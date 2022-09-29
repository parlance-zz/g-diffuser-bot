
import os, gc, warnings
import torch

import numpy as np
import PIL
from types import SimpleNamespace as SN

from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.configuration_utils import FrozenDict

import generation_pb2

from sdgrpcserver.pipeline.unified_pipeline import UnifiedPipeline
from sdgrpcserver.pipeline.safety_checkers import FlagOnlySafetyChecker

from sdgrpcserver.pipeline.scheduling_ddim import DDIMScheduler
from sdgrpcserver.pipeline.scheduling_euler_discrete import EulerDiscreteScheduler
from sdgrpcserver.pipeline.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler

class WithNoop(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

class ProgressBarWrapper(object):

    class InternalTqdm(tqdm):
        def __init__(self, progress_callback, stop_event, iterable):
            self._progress_callback = progress_callback
            self._stop_event = stop_event
            super().__init__(iterable)

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

    def __init__(self, progress_callback, stop_event):
        self._progress_callback = progress_callback
        self._stop_event = stop_event

    def __call__(self, iterable):
        return ProgressBarWrapper.InternalTqdm(self._progress_callback, self._stop_event, iterable)
    

class PipelineWrapper(object):

    def __init__(self, id, device, pipeline, fp16_pipeline=None, vramO=0):
        self._id = id
        self._device = device
        self._vramO = vramO

        self._pipeline = fp16_pipeline if self.fp16 and fp16_pipeline else pipeline
        if self.cuda_only_unet: self._pipeline.unet = fp16_pipeline.unet if fp16_pipeline else self._pipeline.to(torch.float16)

        print(self.fp16, self.attention_slice, self.cuda_only_unet)

        self._plms = pipeline.scheduler
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

    def progress(self, **kwargs):
        print(repr(kwargs))

    def _prepScheduler(self, scheduler):
        scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            warnings.warn(
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file",
                DeprecationWarning,
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        return scheduler

    @property
    def id(self): return self._id

    @property
    def device(self): return self._device

    # -- A few different adjustments we make based on vramO

    @property
    def attention_slice(self): return self.device == "cuda" and self._vramO > 0

    @property
    def fp16(self): return self.device == "cuda" and self._vramO > 1 and not self.cuda_only_unet

    @property
    def cuda_only_unet(self): return self.device == "cuda" and self._vramO > 2


    def activate(self):
        if self.attention_slice: self._pipeline.enable_attention_slicing(1)

        # Pipeline.to is in-place, so we move to the device on activate, and out again on deactivate
        if self.cuda_only_unet: self._pipeline.unet.to(torch.device("cuda"))
        else: self._pipeline.to(self._device)
        
    def deactivate(self):
        self._pipeline.to("cpu")
        if self._device == "cuda": torch.cuda.empty_cache()

    def _autocast(self):
        if self._device == "cuda": return torch.autocast(self._device)
        return WithNoop()

    def generate(self, text, params, image=None, mask=None, negative_text=None, progress_callback=None, stop_event=None):
        generator=None

        if params.seed > 0:
            latents_device = "cpu" if self._pipeline.device.type == "mps" else self._pipeline.device
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
        else:
            raise NotImplementedError("Scheduler not implemented")

        self._pipeline.scheduler = scheduler
        self._pipeline.progress_bar = ProgressBarWrapper(progress_callback, stop_event)

        with self._autocast():
            images = self._pipeline(
                prompt=text,
                negative_prompt=negative_text if negative_text else None,
                init_image=image,
                mask_image=mask,
                strength=params.strength,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg_scale,
                eta=params.eta,
                generator=generator,
                return_dict=False
            )

        return images

#class EngineManager(object):

    def __init__(self, engines=None, models_root="./models", enable_mps=False, vram_optimisation_level=0, nsfw_behaviour="block"):
        assert(engines)
        self.engines = engines
        self._default = None
        self._pipelines = {}
        self._activeId = None
        self._active = None

        self._weight_root = models_root
        self._vramO = vram_optimisation_level
        self._nsfw = nsfw_behaviour

        self._hasCuda = getattr(torch, 'cuda', False) and torch.cuda.is_available()
        self._hasMps = enable_mps and getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available()
        self._device = "cuda" if self._hasCuda else "mps" if self._hasMps else "cpu"

        self._token=os.environ.get("HF_API_TOKEN", True)

        self.loadPipelines()
    
    def _getWeightPath(self, remote_path, local_path):
        if local_path:
            test_path = local_path if os.path.isabs(local_path) else os.path.join(self._weight_root, local_path)
            test_path = os.path.normpath(test_path)
            if os.path.isdir(test_path): return test_path
        return remote_path

    def buildPipeline(self, engine):
        weight_path=self._getWeightPath(engine["model"], engine.get("local_model", None))
        fp16_weight_path=self._getWeightPath(engine["model"], engine.get("local_fp16_model", None))

        use_auth_token=self._token if engine.get("use_auth_token", False) else False

        extra_kwargs={}

        if self._nsfw == "flag":
            extra_kwargs["safety_checker"]=FlagOnlySafetyChecker.from_pretrained(weight_path, subfolder="safety_checker", use_auth_token=use_auth_token)

        if engine["class"] == "StableDiffusionPipeline":
            return PipelineWrapper(
                id=engine["id"], 
                device=self._device, 
                vramO=self._vramO, 
                pipeline=StableDiffusionPipeline.from_pretrained(
                    weight_path,
                    use_auth_token=use_auth_token,
                    **extra_kwargs                        
                ),
                fp16_pipeline=StableDiffusionPipeline.from_pretrained(
                    fp16_weight_path, 
                    revision="fp16", 
                    torch_dtype=torch.float16, 
                    use_auth_token=use_auth_token,
                    **extra_kwargs                        
                ) if self._hasCuda and engine.get("has_fp16", False) else None
            )
        elif engine["class"] == "UnifiedPipeline":
            return PipelineWrapper(
                id=engine["id"], 
                device=self._device, 
                vramO=self._vramO, 
                pipeline=UnifiedPipeline.from_pretrained(
                    weight_path, 
                    use_auth_token=use_auth_token,
                    **extra_kwargs                        
                ),
                fp16_pipeline=UnifiedPipeline.from_pretrained(
                    fp16_weight_path, 
                    revision="fp16", 
                    torch_dtype=torch.float16, 
                    use_auth_token=use_auth_token,
                    **extra_kwargs                        
                ) if self._hasCuda and engine.get("has_fp16", False) else None
            )
    
    def loadPipelines(self):
        for engine in self.engines:
            if not engine.get("enabled", False): continue

            pipe=self.buildPipeline(engine)

            if pipe:
                self._pipelines[pipe.id] = pipe
                if engine.get("default", False): self._default = pipe
            else:
                raise Exception(f'Unknown engine class "{engine["class"]}"')

    def getPipe(self, id):
        """
        Get and activate a pipeline
        TODO: Better activate / deactivate logic. Right now we just keep a max of one pipeline active.
        """

        # If we're already active, just return it
        if self._active and id == self._active.id: return self._active

        # Otherwise deactivate it
        if self._active: self._active.deactivate()

        self._active = self._pipelines[id]
        self._active.activate()

        return self._active
            


