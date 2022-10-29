from functools import cache
import os, warnings, traceback, math, json
from fnmatch import fnmatch
from types import SimpleNamespace as SN

print("/bin/micromamba -r /env -n sd-grpc-server run pip install nltk")
os.system("/bin/micromamba -r /env -n sd-grpc-server run pip install nltk")
print("/bin/micromamba -r /env -n sd-grpc-server run pip install stanza")
os.system("/bin/micromamba -r /env -n sd-grpc-server run pip install stanza")


import torch
import huggingface_hub

from tqdm.auto import tqdm

from transformers import CLIPFeatureExtractor, CLIPModel
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate, logging

import generation_pb2

from sdgrpcserver.pipeline.unified_pipeline import UnifiedPipeline
from sdgrpcserver.pipeline.safety_checkers import FlagOnlySafetyChecker

from sdgrpcserver.pipeline.schedulers.scheduling_ddim import DDIMScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_utils import OldSchedulerMixin
from sdgrpcserver.pipeline.old_schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_dpm2_discrete import DPM2DiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_dpm2_ancestral_discrete import DPM2AncestralDiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_heun_discrete import HeunDiscreteScheduler


class ProgressBarWrapper(object):

    class InternalTqdm(tqdm):
        def __init__(self, progress_callback, stop_event, suppress_output, iterable):
            self._progress_callback = progress_callback
            self._stop_event = stop_event
            super().__init__(iterable, disable=suppress_output)

        def update(self, n=1):
            displayed = super().update(n)
            if displayed and self._progress_callback: self._progress_callback(**self.format_dict)
            return displayed

        def __iter__(self):
            for x in super().__iter__():
                if self._stop_event and self._stop_event.is_set(): 
                    self.set_description("ABORTED")
                    break
                yield x

    def __init__(self, progress_callback, stop_event, suppress_output=False):
        self._progress_callback = progress_callback
        self._stop_event = stop_event
        self._suppress_output = suppress_output

    def __call__(self, iterable):
        return ProgressBarWrapper.InternalTqdm(self._progress_callback, self._stop_event, self._suppress_output, iterable)
    

class EngineMode(object):
    def __init__(self, vram_optimisation_level=0, enable_cuda = True, enable_mps = False):
        self._vramO = vram_optimisation_level
        self._enable_cuda = enable_cuda
        self._enable_mps = enable_mps
    
    @property
    def device(self):
        self._hasCuda = self._enable_cuda and getattr(torch, 'cuda', False) and torch.cuda.is_available()
        self._hasMps = self._enable_mps and getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available()
        return "cuda" if self._hasCuda else "mps" if self._hasMps else "cpu"

    @property
    def attention_slice(self):
        return self.device == "cuda" and self._vramO > 0

    @property
    def fp16(self):
        return self.device == "cuda" and self._vramO > 1

    @property
    def module_mode(self):
        return "one" if self.device == "cuda" and self._vramO > 2 else "all"


class BatchMode:
    def __init__(self, autodetect=False, points=None, simplemax=1, safety_margin=0.2):
        self.autodetect = autodetect
        self.points = json.loads(points) if isinstance(points, str) else points
        self.simplemax = simplemax
        self.safety_margin = safety_margin
    
    def batchmax(self, pixels):
        if self.points:
            # If pixels less than first point, return that max
            if pixels <= self.points[0][0]: return self.points[0][1]

            # Linear interpolate between bracketing points
            pairs = zip(self.points[:-1], self.points[1:])
            for pair in pairs:
                if pixels >= pair[0][0] and pixels <= pair[1][0]:
                    i = (pixels - pair[0][0]) / (pair[1][0] - pair[0][0])
                    return math.floor(pair[0][1] + i * (pair[1][1] - pair[0][1]))

            # Off top of points - assume max of 1
            return 1
        
        if self.simplemax is not None:
            return self.simplemax

        return 1

    def run_autodetect(self, manager, resmax=2048, resstep=256):
        torch.cuda.set_per_process_memory_fraction(1-self.safety_margin)

        pipe = manager.getPipe()
        params = SN(height=512, width=512, cfg_scale=7.5, sampler=generation_pb2.SAMPLER_DDIM, eta=0, steps=8, strength=1, seed=-1)

        l = 32 # Starting value - 512x512 fails inside PyTorch at 32, no amount of VRAM can help

        pixels=[]
        batchmax=[]

        for x in range(512, resmax, resstep):
            params.width = x
            print(f"Determining max batch for {x}")
            # Quick binary search
            r = l # Start with the max from the previous run
            l = 1

            while l < r-1:
                b = (l+r)//2;
                print (f"Trying {b}")
                try:
                    pipe.generate(["A Crocodile"]*b, params, suppress_output=True)
                except Exception as e:
                    r = b
                else:
                    l = b
            
            print (f"Max for {x} is {l}")

            pixels.append(params.width * params.height)
            batchmax.append(l)

            if l == 1:
                print(f"Max res is {x}x512")
                break
            

        self.points=list(zip(pixels, batchmax))
        print("To save these for next time, use these for batch_points:", json.dumps(self.points))

        torch.cuda.set_per_process_memory_fraction(1.0)


class PipelineWrapper(object):

    def __init__(self, id, mode, pipeline):
        self._id = id
        self._mode = mode

        self._pipeline = pipeline

        self._pipeline.enable_attention_slicing(1 if self.mode.attention_slice else None)
        self._pipeline.set_module_mode(self.mode.module_mode)

        self._plms = self._prepScheduler(PNDMScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                skip_prk_steps=True
        ))
        self._klms = self._prepScheduler(LMSDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._ddim = self._prepScheduler(DDIMScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear", 
                clip_sample=False, 
                set_alpha_to_one=False
            ))
        self._euler = self._prepScheduler(EulerDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._eulera = self._prepScheduler(EulerAncestralDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._dpm2 = self._prepScheduler(DPM2DiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._dpm2a = self._prepScheduler(DPM2AncestralDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._heun = self._prepScheduler(HeunDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))

    def _prepScheduler(self, scheduler):
        if isinstance(scheduler, OldSchedulerMixin):
            scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        return scheduler

    @property
    def id(self): return self._id

    @property
    def mode(self): return self._mode

    def activate(self):
        # Pipeline.to is in-place, so we move to the device on activate, and out again on deactivate
        self._pipeline.to(self.mode.device)
        
    def deactivate(self):
        self._pipeline.to("cpu", forceAll=True)
        if self.mode.device == "cuda": torch.cuda.empty_cache()

    def generate(
        self, 
        tokens, 
        params, 
        negative_tokens=None, 
        image=None, 
        mask=None, 
        outmask=None,
        num_images_per_prompt=1,
        progress_callback=None, 
        stop_event=None, 
        suppress_output=False
    ):
        generator=None

        latents_device = "cpu" if self._pipeline.device.type == "mps" else self._pipeline.device

        if isinstance(params.seed, list):
            generator = [torch.Generator(latents_device).manual_seed(seed) for seed in params.seed] 
        elif params.seed > 0:
            generator = torch.Generator(latents_device).manual_seed(params.seed)

        if params.sampler is None or params.sampler == generation_pb2.SAMPLER_DDPM:
            scheduler=self._plms
        elif params.sampler == generation_pb2.SAMPLER_K_LMS:
            scheduler=self._klms
        elif params.sampler == generation_pb2.SAMPLER_DDIM:
            scheduler=self._ddim
        elif params.sampler == generation_pb2.SAMPLER_K_EULER:
            scheduler=self._euler
        elif params.sampler == generation_pb2.SAMPLER_K_EULER_ANCESTRAL:
            scheduler=self._eulera
        elif params.sampler == generation_pb2.SAMPLER_K_DPM_2:
            scheduler=self._dpm2
        elif params.sampler == generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL:
            scheduler=self._dpm2a
        elif params.sampler == generation_pb2.SAMPLER_K_HEUN:
            scheduler=self._heun
        else:
            raise NotImplementedError("Scheduler not implemented")

        self._pipeline.scheduler = scheduler
        self._pipeline.progress_bar = ProgressBarWrapper(progress_callback, stop_event, suppress_output)

        images = self._pipeline(
            prompt=tokens,
            negative_prompt=negative_tokens if negative_tokens else None,
            num_images_per_prompt=num_images_per_prompt,
            init_image=image,
            mask_image=mask,
            outmask_image=outmask,
            strength=params.strength,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg_scale,
            eta=params.eta,
            generator=generator,
            output_type="tensor",
            return_dict=False
        )

        return images

class EngineManager(object):

    def __init__(self, engines, weight_root="./weights", refresh_models=None, mode=EngineMode(), nsfw_behaviour="block", batchMode=BatchMode()):
        self.engines = engines
        self._default = None

        self._pipelines = {}
        self._internal_pipelines = {}

        self._activeId = None
        self._active = None

        self._weight_root = weight_root
        self._refresh_models = refresh_models

        self._mode = mode
        self._batchMode = batchMode
        self._nsfw = nsfw_behaviour
        self._token = os.environ.get("HF_API_TOKEN", True)


    @property
    def mode(self): return self._mode

    @property
    def batchMode(self): return self._batchMode

    def _getWeightPath(self, opts):
        usefp16 = self.mode.fp16 and opts.get("has_fp16", True)

        local_path = opts.get("local_model_fp16" if usefp16 else "local_model", None)
        model_path = opts.get("model", None)
        subfolder = opts.get("subfolder", None)
        use_auth_token=self._token if opts.get("use_auth_token", False) else False

        # Keep a list of the things we tried that failed, so we can report them all in one go later
        # in the case that we weren't able to load it any way at all
        failures = ["Loading model failed, because:"]

        if local_path:
            test_path = local_path if os.path.isabs(local_path) else os.path.join(self._weight_root, local_path)
            test_path = os.path.normpath(test_path)
            if os.path.isdir(test_path): 
                return test_path
            else:
                failures.append(f"    - Local path '{test_path}' doesn't exist")
        else:
            failures.append("    - No local path for " + ("fp16" if usefp16 else "fp32") + " model was provided")
            

        if model_path:
            # We always download the file ourselves, rather than passing model path through to from_pretrained
            # This lets us control the behaviour if the model fails to download
            extra_kwargs={}
            if subfolder: extra_kwargs["allow_patterns"]=f"{subfolder}*"
            if use_auth_token: extra_kwargs["use_auth_token"]=use_auth_token

            cache_path = None
            attempt_download = False

            try:
                # Try getting the cached path without connecting to internet
                cache_path = huggingface_hub.snapshot_download(
                    model_path, 
                    repo_type="model", 
                    local_files_only=True,
                    **extra_kwargs
                )
            except FileNotFoundError:
                attempt_download = True
            except:
                failures.append("    - Couldn't query local HuggingFace cache. Error was:")
                failures.append(traceback.format_exc())

            if self._refresh_models:
                attempt_download = attempt_download or any((True for pattern in self._refresh_models if fnmatch(model_path, pattern)))

            if attempt_download:
                try:
                    cache_path = huggingface_hub.snapshot_download(
                        model_path, 
                        repo_type="model", 
                        **extra_kwargs
                    )
                except:
                    if cache_path: 
                        print(f"Couldn't refresh cache for {model_path}. Using existing cache.")
                    else:
                        failures.append("    - Downloading from HuggingFace failed. Error was:")
                        failures.append(traceback.format_exc())
            
            if cache_path: return cache_path

        else:
            failures.append("    - No remote model name was provided")

        raise EnvironmentError("\n".join(failures))

    def fromPretrained(self, klass, opts, extra_kwargs = None):
        if extra_kwargs is None: extra_kwargs = {}

        use_auth_token=self._token if opts.get("use_auth_token", False) else False

        weight_path=self._getWeightPath(opts)
        if self.mode.fp16: extra_kwargs["torch_dtype"]=torch.float16
        if opts.get('subfolder', None): extra_kwargs['subfolder'] = opts.get('subfolder')

        # Supress warnings during pipeline load. Unfortunately there isn't a better 
        # way to override the behaviour (beyond duplicating a huge function)
        current_log_level = logging.get_verbosity()
        logging.set_verbosity(logging.ERROR)

        result = klass.from_pretrained(weight_path, use_auth_token=use_auth_token, **extra_kwargs)

        logging.set_verbosity(current_log_level)

        return result

    def buildPipeline(self, engine):
        extra_kwargs={}

        borrow = engine.get("borrow", None)
        if borrow:
            for lender, modules in borrow.items():
                pipeline = self._internal_pipelines[lender]
                for module in modules: extra_kwargs[module] = getattr(pipeline, module)

        if self._nsfw == "flag":
            extra_kwargs["safety_checker"] = self.fromPretrained(FlagOnlySafetyChecker, {**engine, "subfolder": "safety_checker"})
        elif self._nsfw == "ignore":
            extra_kwargs["safety_checker"] = None

        for name, opts in engine.get("overrides", {}).items():
            if name == "vae":
                extra_kwargs["vae"] = self.fromPretrained(AutoencoderKL, opts)
            elif name == "inpaint_unet":
                extra_kwargs["inpaint_unet"] = self.fromPretrained(UNet2DConditionModel, {**opts, "subfolder": "unet"})
            elif name == "clip_model":
                extra_kwargs["clip_model"] = self.fromPretrained(CLIPModel, opts)
                extra_kwargs["feature_extractor"] = self.fromPretrained(CLIPFeatureExtractor, opts)
        
        pipeline = None

        if engine["class"] == "StableDiffusionPipeline":
           pipeline = self.fromPretrained(StableDiffusionPipeline, engine, extra_kwargs)

        elif engine["class"] == "UnifiedPipeline":
            if "inpaint_unet" not in extra_kwargs: 
                extra_kwargs["inpaint_unet"] = None
            if "clip_model" not in extra_kwargs: 
                extra_kwargs["clip_model"] = None
            
            pipeline = self.fromPretrained(UnifiedPipeline, engine, extra_kwargs)

        else:
            raise Exception(f'Unknown engine class "{engine["class"]}"')

        if engine.get("options", False):
            try:
                pipeline.set_options(engine.get("options"))
            except:
                raise ValueError(f"Engine {engine['id']} has options, but created pipeline rejected them")
        
        return pipeline

    def loadPipelines(self):

        print("Loading engines...")

        for engine in self.engines:
            if not engine.get("enabled", False): continue

            engineid = engine["id"]
            if engine.get("default", False): self._default = engineid

            print("  -"+engineid+"...")

            self._internal_pipelines[engineid] = pipeline = self.buildPipeline(engine)

            self._pipelines[engineid] = PipelineWrapper(
                id=engineid,
                mode=self._mode,
                pipeline=pipeline
            )

        if self.batchMode.autodetect:
            self.batchMode.run_autodetect(self)


    def getStatus(self):
        return {engine["id"]: engine["id"] in self._pipelines for engine in self.engines if engine.get("enabled", True)}

    def getPipe(self, id=None):
        """
        Get and activate a pipeline
        TODO: Better activate / deactivate logic. Right now we just keep a max of one pipeline active.
        """

        if id is None: id = self._default

        # If we're already active, just return it
        if self._active and id == self._active.id: return self._active

        # Otherwise deactivate it
        if self._active: self._active.deactivate()

        self._active = self._pipelines[id]
        self._active.activate()

        return self._active
            


