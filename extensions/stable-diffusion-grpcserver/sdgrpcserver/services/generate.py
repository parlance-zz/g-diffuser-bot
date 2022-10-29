
from math import sqrt
import random, traceback, threading
from types import SimpleNamespace as SN
import torch

import grpc
import generation_pb2, generation_pb2_grpc

from sdgrpcserver.utils import image_to_artifact, artifact_to_image

from sdgrpcserver import images
from sdgrpcserver.debug_recorder import DebugNullRecorder

def buildDefaultMaskPostAdjustments():
    hardenMask = generation_pb2.ImageAdjustment()
    hardenMask.levels.input_low = 0
    hardenMask.levels.input_high = 0.05
    hardenMask.levels.output_low = 0
    hardenMask.levels.output_high = 1

    blur = generation_pb2.ImageAdjustment()
    blur.blur.sigma = 32
    blur.blur.direction = generation_pb2.DIRECTION_UP

    #levels = generation_pb2.ImageAdjustment()
    #levels.levels.input_low = 0
    #levels.levels.input_high = 0.5
    #levels.levels.output_low = 0
    #levels.levels.output_high = 1

    return [hardenMask, blur] #, levels]

defaultMaskPostAdjustments = buildDefaultMaskPostAdjustments();

debugCtr=0

class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, manager, debug_recorder = DebugNullRecorder()):
        self._manager = manager
        self._debug_recorder = debug_recorder

    def saveDebugTensor(self, tensor):
        global debugCtr
        debugCtr += 1 
        with open(f"debug-{debugCtr}.png", "wb") as f:
            f.write(images.toPngBytes(tensor)[0])


    def unimp(self, what):
        raise NotImplementedError(f"{what} not implemented")

    def _handleImageAdjustment(self, tensor, adjustments):
        if type(tensor) is bytes: tensor = images.fromPngBytes(tensor)

        #self.saveDebugTensor(tensor)

        for adjustment in adjustments:
            which = adjustment.WhichOneof("adjustment")

            if which == "blur":
                sigma = adjustment.blur.sigma
                direction = adjustment.blur.direction

                if direction == generation_pb2.DIRECTION_DOWN or direction == generation_pb2.DIRECTION_UP:
                    orig = tensor
                    repeatCount=256
                    sigma /= sqrt(repeatCount)

                    for _ in range(repeatCount):
                        tensor = images.gaussianblur(tensor, sigma)
                        if direction == generation_pb2.DIRECTION_DOWN:
                            tensor = torch.minimum(tensor, orig)
                        else:
                            tensor = torch.maximum(tensor, orig)
                else:
                    tensor = images.gaussianblur(tensor, adjustment.blur.sigma)
            elif which == "invert":
                tensor = images.invert(tensor)
            elif which == "levels":
                tensor = images.levels(tensor, adjustment.levels.input_low, adjustment.levels.input_high, adjustment.levels.output_low, adjustment.levels.output_high)
            elif which == "channels":
                tensor = images.channelmap(tensor, [adjustment.channels.r,  adjustment.channels.g,  adjustment.channels.b,  adjustment.channels.a])
            elif which == "rescale":
                self.unimp("Rescale")
            elif which == "crop":
                tensor = images.crop(tensor, adjustment.crop.top, adjustment.crop.left, adjustment.crop.height, adjustment.crop.width)
            
            #self.saveDebugTensor(tensor)
        
        return tensor

    def Generate(self, request, context):
        with self._debug_recorder.record(request.request_id) as recorder:
            recorder.store('generate request', request)

            try:
                # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
                if request.requested_type != generation_pb2.ARTIFACT_NONE and request.requested_type != generation_pb2.ARTIFACT_IMAGE:
                    self.unimp('Generation of anything except images')

                # Basic parameters
                params=SN(
                    height=512,
                    width=512,
                    cfg_scale=7.5,
                    eta=0,
                    sampler=None,
                    steps=50,
                    seed=-1,
                    samples=1,
                    strength=0.8
                )

                for field in vars(params):
                    try:
                        if request.image.HasField(field):
                            setattr(params, field, getattr(request.image, field))
                    except Exception as e:
                        pass

                # Extract prompt inputs
                image=None
                inMask=None
                outMask=None
                tokens=[]
                negative=[]

                for prompt in request.prompt:
                    which = prompt.WhichOneof("prompt")
                    if which == "text": 
                        weight = 1.0
                        if prompt.HasField("parameters") and prompt.parameters.HasField("weight"): weight = prompt.parameters.weight
                        if weight > 0: tokens.append((prompt.text, weight))
                        else: negative.append((prompt.text, -weight))
                    elif which == "tokens": 
                        self.unimp("Token prompts")
                    else:
                        if prompt.artifact.type == generation_pb2.ARTIFACT_IMAGE:
                            image = images.fromPngBytes(prompt.artifact.binary).to(self._manager.mode.device)
                            image = self._handleImageAdjustment(image, prompt.artifact.adjustments)
                            params.height = image.shape[2]
                            params.width = image.shape[3]
                        elif prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                            mask = images.fromPngBytes(prompt.artifact.binary).to(self._manager.mode.device)
                            inMask = self._handleImageAdjustment(mask, prompt.artifact.adjustments)

                            postAdjustments = prompt.artifact.postAdjustments
                            if not postAdjustments: postAdjustments = defaultMaskPostAdjustments

                            outMask = self._handleImageAdjustment(inMask, postAdjustments)
                        else:
                            self.unimp(f"Artifact prompts of type {prompt.artifact.type}")

                
                seeds = list(request.image.seed)

                for extras in request.image.parameters:
                    if extras.HasField("sampler"):
                        if extras.sampler.HasField("cfg_scale"): params.cfg_scale = extras.sampler.cfg_scale
                        if extras.sampler.HasField("eta"): params.eta = extras.sampler.eta
                    if extras.HasField("schedule"):
                        if extras.schedule.HasField("start"): params.strength = extras.schedule.start            
                
                if request.image.HasField("transform") and request.image.transform.WhichOneof("type") == "diffusion": params.sampler = request.image.transform.diffusion

                try:
                    pipe = self._manager.getPipe(request.engine_id)
                except KeyError as e:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("Engine not found")
                    return

                stop_event = threading.Event()
                context.add_callback(lambda: stop_event.set())

                ctr = 0
                batchmax = self._manager.batchMode.batchmax(params.width * params.height)

                # If we weren't given any seeds at all, just start with a single -1
                if not seeds: seeds = [-1]

                # Replace any negative seeds with a randomly selected one
                seeds = [seed if seed >= 0 else random.randrange(0, 2**32-1) for seed in seeds]

                # Fill seeds up to params.samples if we didn't get passed enough
                if len(seeds) < params.samples:
                    # Starting with the last seed we were given
                    nextseed = seeds[-1]+1
                    while len(seeds) < params.samples: 
                        seeds.append(nextseed)
                        nextseed += 1

                # Calculate the most even possible split across batchmax
                if params.samples <= batchmax:
                    batches = [params.samples]
                elif params.samples % batchmax == 0:
                    batches = [batchmax] * (params.samples // batchmax)
                else:
                    d = params.samples // batchmax + 1
                    batchsize = params.samples // d
                    r = params.samples - batchsize * d
                    batches = [batchsize+1]*r + [batchsize] * (d-r)

                # Loop until we've returned the requested number of images
                for batch in batches:
                    params.seed, seeds = seeds[:batch], seeds[batch:]

                    print(f'Generating {repr(params)}, {"with Image" if image != None else ""}, {"with Mask" if inMask != None else ""}')

                    args={
                        "tokens": tokens,
                        "negative_tokens": negative if negative else None,
                        "num_images_per_prompt": batch,
                        "image": image,
                        "mask": inMask,
                        "outmask": outMask,
                        "params": params,
                    }

                    recorder.store('pipe.generate calls', args)

                    results = pipe.generate(**args, stop_event=stop_event)

                    for i, (result_image, nsfw) in enumerate(zip(results[0], results[1])):
                        answer = generation_pb2.Answer()
                        answer.request_id=request.request_id
                        answer.answer_id=f"{request.request_id}-{ctr}"
                        artifact=image_to_artifact(result_image)
                        artifact.finish_reason=generation_pb2.FILTER if nsfw else generation_pb2.NULL
                        artifact.index=ctr
                        artifact.seed=params.seed[i]
                        answer.artifacts.append(artifact)

                        recorder.store('pipe.generate result', artifact)
                        
                        yield answer
                        ctr += 1
                
            except NotImplementedError as e:
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details(str(e))
                print(f"Unsupported request parameters: {e}")
            except Exception as e:
                traceback.print_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Something went wrong")
