
import os, warnings
from sdgrpcserver.pipeline.old_schedulers.scheduling_utils import OldSchedulerMixin
import torch

from tqdm.auto import tqdm

# Patch attention to use MemoryEfficientCrossAttention if xformers is available
from diffusers.models import attention
from sdgrpcserver.pipeline.fastattention import has_xformers, MemoryEfficientCrossAttention
print(f"Using xformers: {'yes' if has_xformers() else 'no'}")
if has_xformers(): attention.CrossAttention = MemoryEfficientCrossAttention

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate
from diffusers.models import AutoencoderKL

import generation_pb2

from sdgrpcserver.pipeline.unified_pipeline import UnifiedPipeline
from sdgrpcserver.pipeline.safety_checkers import FlagOnlySafetyChecker

from sdgrpcserver.pipeline.schedulers.scheduling_ddim import DDIMScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_dpm2_discrete import DPM2DiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_dpm2_ancestral_discrete import DPM2AncestralDiscreteScheduler
from sdgrpcserver.pipeline.old_schedulers.scheduling_heun_discrete import HeunDiscreteScheduler

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

    def generate(self, text, params, image=None, mask=None, outmask=None, negative_text=None, progress_callback=None, stop_event=None):
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
        elif params.sampler == generation_pb2.SAMPLER_K_DPM_2:
            scheduler=self._dpm2
        elif params.sampler == generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL:
            scheduler=self._dpm2a
        elif params.sampler == generation_pb2.SAMPLER_K_HEUN:
            scheduler=self._heun
        else:
            raise NotImplementedError("Scheduler not implemented")

        self._pipeline.scheduler = scheduler
        self._pipeline.progress_bar = ProgressBarWrapper(progress_callback, stop_event)

        images = self._pipeline(
            prompt=text,
            negative_prompt=negative_text if negative_text else None,
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

    def __init__(self, engines, weight_root="./weights", mode=EngineMode(), nsfw_behaviour="block"):
        self.engines = engines
        self._default = None
        self._pipelines = {}
        self._activeId = None
        self._active = None

        self._weight_root = weight_root

        self._mode = mode
        self._nsfw = nsfw_behaviour
        self._token = os.environ.get("HF_API_TOKEN", True)

    @property
    def mode(self): return self._mode

    def _getWeightPath(self, remote_path, local_path):
        if local_path:
            test_path = local_path if os.path.isabs(local_path) else os.path.join(self._weight_root, local_path)
            test_path = os.path.normpath(test_path)
            if os.path.isdir(test_path): return test_path
        return remote_path

    def buildPipeline(self, engine):
        if self.mode.fp16:
           weight_path=self._getWeightPath(engine["model"], engine.get("local_model_fp16", None))
        else:
           weight_path=self._getWeightPath(engine["model"], engine.get("local_model", None))


        use_auth_token=self._token if engine.get("use_auth_token", False) else False

        extra_kwargs={}

        if self.mode.fp16:
            extra_kwargs["revision"]="fp16"
            extra_kwargs["torch_dtype"]=torch.float16

        if self._nsfw == "flag":
            extra_kwargs["safety_checker"]=FlagOnlySafetyChecker.from_pretrained(
                weight_path, 
                subfolder="safety_checker", 
                use_auth_token=use_auth_token,
                **extra_kwargs
            )

        if engine["class"] == "StableDiffusionPipeline":
            return PipelineWrapper(
                id=engine["id"],
                mode=self._mode,
                pipeline=StableDiffusionPipeline.from_pretrained(
                    weight_path,
                    use_auth_token=use_auth_token,
                    **extra_kwargs                        
                )
            )
        elif engine["class"] == "UnifiedPipeline":
            vae_extra_kwargs = {}
            if self.mode.fp16: vae_extra_kwargs["torch_dtype"]=torch.float16
            return PipelineWrapper(
                id=engine["id"],
                mode=self._mode,
                pipeline=UnifiedPipeline.from_pretrained(
                    weight_path, 
                    use_auth_token=use_auth_token,
                    vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", **vae_extra_kwargs),
                    **extra_kwargs                        
                )
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

    def getStatus(self):
        return {engine["id"]: engine["id"] in self._pipelines for engine in self.engines if engine.get("enabled", True)}

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
            


