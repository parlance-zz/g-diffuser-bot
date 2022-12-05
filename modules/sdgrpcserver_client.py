#!/bin/which python3

# Modified version of Stability-AI SDK client.py. Changes:
#   - Calls cancel on ctrl-c to allow server to abort
#   - Supports setting ETA parameter
#   - Supports actually setting CLIP guidance strength
#   - Supports negative prompt by setting a prompt with negative weight
#   - Supports sending key to machines on local network over HTTP (not HTTPS)

import sys
import os
import uuid
import random
import io
import logging
import time
import mimetypes
#import signal
from sdgrpcserver.sonora import client as sonora_client

import grpc
from argparse import ArgumentParser, Namespace
from typing import Dict, Generator, List, Optional, Union, Any, Sequence, Tuple
from google.protobuf.json_format import MessageToJson
#from PIL import Image

# this is necessary because of how the auto-generated code constructs its imports
thisPath = os.path.dirname(os.path.abspath(__file__))
genPath = thisPath +"/sdgrpcserver/generated"

sys.path.append(str(genPath))

import generation_pb2 as generation
import generation_pb2_grpc as generation_grpc
import engines_pb2 as engines
import engines_pb2_grpc as engines_grpc

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

SAMPLERS: Dict[str, int] = {
    "ddim": generation.SAMPLER_DDIM,
    "plms": generation.SAMPLER_DDPM,
    "k_euler": generation.SAMPLER_K_EULER,
    "k_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation.SAMPLER_K_HEUN,
    "k_dpm_2": generation.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation.SAMPLER_K_LMS,
    "dpm_fast": generation.SAMPLER_DPM_FAST,
    "dpm_adaptive": generation.SAMPLER_DPM_ADAPTIVE,
    "dpmspp_1": generation.SAMPLER_DPMSOLVERPP_1ORDER,
    "dpmspp_2": generation.SAMPLER_DPMSOLVERPP_2ORDER,
    "dpmspp_3": generation.SAMPLER_DPMSOLVERPP_3ORDER,
    "dpmspp_2s_ancestral": generation.SAMPLER_DPMSOLVERPP_2S_ANCESTRAL,
    "dpmspp_sde": generation.SAMPLER_DPMSOLVERPP_SDE,
    "dpmspp_2m": generation.SAMPLER_DPMSOLVERPP_2M,
}

def get_sampler_from_str(s: str) -> generation.DiffusionSampler:
    """
    Convert a string to a DiffusionSampler enum.

    :param s: The string to convert.
    :return: The DiffusionSampler enum.
    """
    algorithm_key = s.lower().strip()
    algorithm = SAMPLERS.get(algorithm_key, None)
    if algorithm is None:
        raise ValueError(f"unknown sampler {s}")
    
    return algorithm

def image_to_prompt(im, init: bool = False, mask: bool = False) -> generation.Prompt:
    if init and mask:
        raise ValueError("init and mask cannot both be True")
    buf = io.BytesIO(im)
    #buf = io.BytesIO()
    #im.save(buf, format="PNG")
    #buf.seek(0)
    if mask:
        return generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_MASK, binary=buf.getvalue()
            )
        )
    return generation.Prompt(
        artifact=generation.Artifact(
            type=generation.ARTIFACT_IMAGE, binary=buf.getvalue()
        ),
        parameters=generation.PromptParameters(init=init),
    )

def process_artifacts_from_answers(
    prefix: str,
    answers: Union[
        Generator[generation.Answer, None, None], Sequence[generation.Answer]
    ],
    write: bool = True,
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Process the Artifacts from the Answers.

    :param prefix: The prefix for the artifact filenames.
    :param answers: The Answers to process.
    :param write: Whether to write the artifacts to disk.
    :param verbose: Whether to print the artifact filenames.
    :return: A Generator of tuples of artifact filenames and Artifacts, intended
        for passthrough.
    """
    idx = 0
    for resp in answers:
        for artifact in resp.artifacts:
            artifact_p = f"{prefix}-{resp.request_id}-{resp.answer_id}-{idx}"
            if artifact.type == generation.ARTIFACT_IMAGE:
                ext = mimetypes.guess_extension(artifact.mime)
                contents = artifact.binary
            elif artifact.type == generation.ARTIFACT_CLASSIFICATIONS:
                ext = ".pb.json"
                contents = MessageToJson(artifact.classifier).encode("utf-8")
            elif artifact.type == generation.ARTIFACT_TEXT:
                ext = ".pb.json"
                contents = MessageToJson(artifact).encode("utf-8")
            else:
                ext = ".pb"
                contents = artifact.SerializeToString()
            out_p = f"{artifact_p}{ext}"
            if write:
                with open(out_p, "wb") as f:
                    f.write(bytes(contents))
                    if verbose:
                        artifact_t = generation.ArtifactType.Name(artifact.type)
                        logger.info(f"wrote {artifact_t} to {out_p}")
                        if artifact.finish_reason == generation.FILTER: logger.info(f"{artifact_t} flagged as NSFW")

            yield [out_p, artifact]
            idx += 1




class StabilityInference:
    def __init__(
        self,
        host: str = "grpc.stability.ai:443",
        key: str = "",
        engine: str = "stable-diffusion-v1-5",
        verbose: bool = False,
        wait_for_ready: bool = True,
        use_grpc_web: bool = False,
    ):
        """
        Initialize the client.

        :param host: Host to connect to.
        :param key: Key to use for authentication.
        :param engine: Engine to use.
        :param verbose: Whether to print debug messages.
        :param wait_for_ready: Whether to wait for the server to be ready, or
            to fail immediately.
        """
        self.verbose = verbose
        self.engine = engine

        self.grpc_args = {"wait_for_ready": wait_for_ready}

        if verbose:
            logger.info(f"Opening channel to {host}")

        if use_grpc_web:
            channel = sonora_client.insecure_web_channel(host)
            channel._session.headers.update({"authorization": "Bearer {0}".format(key)})
            
        else:
            call_credentials = []

            if key:
                call_credentials.append(grpc.access_token_call_credentials(f"{key}"))
                
                if host.endswith("443"):
                    channel_credentials = grpc.ssl_channel_credentials()
                else:
                    print("Key provided but channel is not HTTPS - assuming a local network")
                    channel_credentials = grpc.local_channel_credentials()
                
                    channel = grpc.secure_channel(
                        host, 
                        grpc.composite_channel_credentials(channel_credentials, *call_credentials)
                    )
            else:
                channel = grpc.insecure_channel(host)

        if verbose:
            logger.info(f"Channel opened to {host}")
        self.stub = generation_grpc.GenerationServiceStub(channel)
        self.engines_stub = engines_grpc.EnginesServiceStub(channel)
        return

    def list_engines(self):
        rq = engines.ListEnginesRequest()
        _engines = self.engines_stub.ListEngines(rq, **self.grpc_args)
        engines_list = []
        for i in range(len(_engines.engine)):
            _engine = { "id": str(_engines.engine[i].id),
                        "name": str(_engines.engine[i].name),
                        "description": str(_engines.engine[i].description),
                        "ready": bool(_engines.engine[i].ready),
                      }
            engines_list.append(_engine)
        return engines_list

    def generate(
        self,
        prompt: Union[str, List[str], generation.Prompt, List[generation.Prompt]],
        negative_prompt: str = None,
        init_image = None,
        mask_image = None,
        height: int = 512,
        width: int = 512,
        start_schedule: float = 1.0,
        end_schedule: float = 0.01,
        cfg_scale: float = 7.0,
        eta: float = 0.0,
        sampler: generation.DiffusionSampler = generation.SAMPLER_K_LMS,
        steps: int = 50,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        safety: bool = True,
        classifiers: Optional[generation.ClassifierParameters] = None,
        guidance_preset: generation.GuidancePreset = generation.GUIDANCE_PRESET_NONE,
        guidance_cuts: int = 0,
        guidance_strength: Optional[float] = None,
        guidance_prompt: Union[str, generation.Prompt] = None,
        guidance_models: List[str] = None,
    ) -> Generator[generation.Answer, None, None]:
        """
        Generate images from a prompt.

        :param prompt: Prompt to generate images from.
        :param init_image: Init image.
        :param mask_image: Mask image
        :param height: Height of the generated images.
        :param width: Width of the generated images.
        :param start_schedule: Start schedule for init image.
        :param end_schedule: End schedule for init image.
        :param cfg_scale: Scale of the configuration.
        :param sampler: Sampler to use.
        :param steps: Number of steps to take.
        :param seed: Seed for the random number generator.
        :param samples: Number of samples to generate.
        :param safety: DEPRECATED/UNUSED - Cannot be disabled.
        :param classifiers: DEPRECATED/UNUSED - Has no effect on image generation.
        :param guidance_preset: Guidance preset to use. See generation.GuidancePreset for supported values.
        :param guidance_cuts: Number of cuts to use for guidance.
        :param guidance_strength: Strength of the guidance. We recommend values in range [0.0,1.0]. A good default is 0.25
        :param guidance_prompt: Prompt to use for guidance, defaults to `prompt` argument (above) if not specified.
        :param guidance_models: Models to use for guidance.
        :return: Generator of Answer objects.
        """
        if (prompt is None) and (init_image is None):
            raise ValueError("prompt and/or init_image must be provided")

        if (mask_image is not None) and (init_image is None):
            raise ValueError("If mask_image is provided, init_image must also be provided")

        if not seed:
            seed = [random.randrange(0, 4294967295)]
        elif isinstance(seed, int):
            seed = [seed]
        else:
            seed = list(seed)

        prompts: List[generation.Prompt] = []
        if any(isinstance(prompt, t) for t in (str, generation.Prompt)):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, str):
                p = generation.Prompt(text=p)
            elif not isinstance(p, generation.Prompt):
                raise TypeError("prompt must be a string or generation.Prompt object")
            prompts.append(p)

        if negative_prompt:
            prompts += [generation.Prompt(
                text=negative_prompt, 
                parameters=generation.PromptParameters(weight=-1)
            )]

        step_parameters = dict(
            scaled_step=0,
            sampler=generation.SamplerParameters(
                cfg_scale=cfg_scale,
                eta=eta,
            ),
        )

        # NB: Specifying schedule when there's no init image causes washed out results
        if init_image is not None:
            step_parameters['schedule'] = generation.ScheduleParameters(
                start=start_schedule,
                end=end_schedule,
            )
            prompts += [image_to_prompt(init_image, init=True)]

            if mask_image is not None:
                prompts += [image_to_prompt(mask_image, mask=True)]
        
        if guidance_prompt:
            if isinstance(guidance_prompt, str):
                guidance_prompt = generation.Prompt(text=guidance_prompt)
            elif not isinstance(guidance_prompt, generation.Prompt):
                raise ValueError("guidance_prompt must be a string or Prompt object")
        if guidance_strength == 0.0:
            guidance_strength = None

        # Build our CLIP parameters
        if guidance_preset is not generation.GUIDANCE_PRESET_NONE:
            # to do: make it so user can override this
            # step_parameters['sampler']=None

            if guidance_models:
                guiders = [generation.Model(alias=model) for model in guidance_models]
            else:
                guiders = None

            if guidance_cuts:
                cutouts = generation.CutoutParameters(count=guidance_cuts)
            else:
                cutouts = None

            step_parameters["guidance"] = generation.GuidanceParameters(
                guidance_preset=guidance_preset,
                instances=[
                    generation.GuidanceInstanceParameters(
                        guidance_strength=guidance_strength,
                        models=guiders,
                        cutouts=cutouts,
                        prompt=guidance_prompt,
                    )
                ],
            )

        image_parameters=generation.ImageParameters(
            transform=generation.TransformType(diffusion=sampler),
            height=height,
            width=width,
            seed=seed,
            steps=steps,
            samples=samples,
            parameters=[generation.StepParameter(**step_parameters)],
        )

        return self.emit_request(prompt=prompts, image_parameters=image_parameters)

    # The motivation here is to facilitate constructing requests by passing protobuf objects directly.
    def emit_request(
        self,
        prompt: generation.Prompt,
        image_parameters: generation.ImageParameters,
        engine_id: str = None,
        request_id: str = None,
    ):
        if not request_id:
            request_id = str(uuid.uuid4())
        if not engine_id:
            engine_id = self.engine
        
        rq = generation.Request(
            engine_id=engine_id,
            request_id=request_id,
            prompt=prompt,
            image=image_parameters
        )
        
        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        answers = self.stub.Generate(rq, **self.grpc_args)

        def cancel_request(unused_signum, unused_frame):
            #print("Cancelling")
            answers.cancel()
            #sys.exit(0)

        #signal.signal(signal.SIGINT, cancel_request)

        for answer in answers:
            duration = time.time() - start
            if self.verbose:
                if len(answer.artifacts) > 0:
                    artifact_ts = [
                        generation.ArtifactType.Name(artifact.type)
                        for artifact in answer.artifacts
                    ]
                    logger.info(
                        f"Got {answer.answer_id} with {artifact_ts} in "
                        f"{duration:0.2f}s"
                    )
                else:
                    logger.info(
                        f"Got keepalive {answer.answer_id} in "
                        f"{duration:0.2f}s"
                    )

            yield answer
            start = time.time()