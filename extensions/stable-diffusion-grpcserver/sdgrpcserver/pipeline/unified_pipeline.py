import inspect, traceback
import time
from mimetypes import init
from typing import Callable, List, Optional, Union

import numpy as np
from sdgrpcserver.pipeline.old_schedulers.scheduling_utils import OldSchedulerMixin
import torch
import torchvision
import torchvision.transforms as T

import PIL
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class UnifiedMode(object):

    def __init__(self, **_):
        self.t_start = 0

    def generateLatents(self):
        raise NotImplementedError('Subclasses must implement')

    def latentStep(self, latents, i, t, steppos):
        return latents

class Txt2imgMode(UnifiedMode):

    def __init__(self, pipeline, generator, height, width, latents_dtype, batch_total, **kwargs):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = pipeline.scheduler

        self.generator = generator

        self.latents_device = "cpu" if self.device.type == "mps" else self.device
        self.latents_dtype = latents_dtype
        self.latents_shape = (
            batch_total, 
            pipeline.unet.in_channels, 
            height // 8, 
            width // 8
        )

    def generateLatents(self):
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents = torch.randn(
            self.latents_shape, 
            generator=self.generator, 
            device=self.latents_device, 
            dtype=self.latents_dtype
        )
        
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        if isinstance(self.scheduler, OldSchedulerMixin): 
            return latents * self.scheduler.sigmas[0]
        else:
            return latents * self.scheduler.init_noise_sigma

    def timestepsTensor(self):
        return super().timestepsTensor(0)


class Img2imgMode(UnifiedMode):

    def __init__(self, pipeline, generator, init_image, latents_dtype, batch_total, num_inference_steps, strength, **kwargs):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        
        super().__init__(**kwargs)

        self.device = pipeline.device
        self.scheduler = pipeline.scheduler
        self.pipeline = pipeline

        self.generator = generator

        self.latents_dtype = latents_dtype
        self.batch_total = batch_total
        
        self.offset = self.scheduler.config.get("steps_offset", 0)
        self.init_timestep = int(num_inference_steps * strength) + self.offset
        self.init_timestep = min(self.init_timestep, num_inference_steps)
        self.t_start = max(num_inference_steps - self.init_timestep + self.offset, 0)

        if isinstance(init_image, PIL.Image.Image):
            self.init_image = self.preprocess(init_image)
        else:
            self.init_image = self.preprocess_tensor(init_image)

    def preprocess(self, image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def preprocess_tensor(self, tensor):
        # Make sure it's BCHW not just CHW
        if tensor.ndim == 3: tensor = tensor[None, ...]
        # Strip any alpha
        tensor = tensor[:, [0,1,2]]
        # Adjust to -1 .. 1
        tensor = 2.0 * tensor - 1.0
        # Done
        return tensor

    def _buildInitialLatents(self):
        init_image = self.init_image.to(device=self.device, dtype=self.latents_dtype)
        init_latent_dist = self.pipeline.vae.encode(init_image).latent_dist
        init_latents = init_latent_dist.sample(generator=self.generator)
        init_latents = 0.18215 * init_latents

        # expand init_latents for batch_size
        return torch.cat([init_latents] * self.batch_total, dim=0)

    def _getSchedulerNoiseTimestep(self, i, t = None):
        """Figure out the timestep to pass to scheduler.add_noise
        If it's an old-style scheduler:
          - return the index as a single integer tensor

        If it's a new-style scheduler:
          - if we know the timestep use it
          - otherwise look up the timestep in the scheduler
          - either way, return a tensor * batch_total on our device
        """
        if isinstance(self.scheduler, OldSchedulerMixin): 
            return torch.tensor(i)
        else:
            timesteps = t if t != None else self.scheduler.timesteps[i]
            return torch.tensor([timesteps] * self.batch_total, device=self.device)

    def _addInitialNoise(self, latents):
        # NOTE: We run K_LMS in float32, because it seems to have problems with float16
        noise_dtype=torch.float32 if isinstance(self.scheduler, LMSDiscreteScheduler) else self.latents_dtype

        self.image_noise = torch.randn(latents.shape, generator=self.generator, device=self.device, dtype=noise_dtype)
        result = self.scheduler.add_noise(latents.to(noise_dtype), self.image_noise, self._getSchedulerNoiseTimestep(self.t_start))
        return result.to(self.latents_dtype) # Old schedulers return float32, and we force K_LMS into float32, but we need to return float16

    def generateLatents(self):
        init_latents = self._buildInitialLatents()
        init_latents = self._addInitialNoise(init_latents)
        return init_latents

class MaskProcessorMixin(object):

    def preprocess_mask(self, mask):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    def preprocess_mask_tensor(self, tensor):
        if tensor.ndim == 3: tensor = tensor[None, ...]
        # Create 4 channels from the R channel
        tensor = tensor[:, [0, 0, 0, 0]]
        # Resize to 1/8th normal
        tensor = T.functional.resize(tensor, [tensor.shape[2]//8, tensor.shape[3]//8], T.InterpolationMode.NEAREST)
        # Invert
        tensor = 1 - tensor
        # Done
        return tensor

class OriginalInpaintMode(Img2imgMode, MaskProcessorMixin):

    def __init__(self, mask_image, **kwargs):
        super().__init__(**kwargs)

        if isinstance(mask_image, PIL.Image.Image):
            self.mask_image = self.preprocess_mask(mask_image)
        else:
            self.mask_image = self.preprocess_mask_tensor(mask_image)

        self.mask = self.mask_image.to(device=self.device, dtype=self.latents_dtype)
        self.mask = torch.cat([self.mask] * self.batch_total)

    def generateLatents(self):
        init_latents = self._buildInitialLatents()

        self.init_latents_orig = init_latents

        init_latents = self._addInitialNoise(init_latents)
        return init_latents

    def latentStep(self, latents, i, t, steppos):
        # masking
        init_latents_proper = self.scheduler.add_noise(self.init_latents_orig, self.image_noise, torch.tensor([t]))
        return (init_latents_proper * self.mask) + (latents * (1 - self.mask))

class EnhancedInpaintMode(Img2imgMode, MaskProcessorMixin):

    def __init__(self, mask_image, num_inference_steps, strength, **kwargs):
        # Check strength
        if strength < 0 or strength > 2:
            raise ValueError(f"The value of strength should in [0.0, 2.0] but is {strength}")

        # When strength > 1, we start allowing the protected area to change too. Remember that and then set strength
        # to 1 for parent class
        self.fill_with_shaped_noise = strength >= 1.0
        self.mask_scale = min(2 - strength, 1)
        strength = min(strength, 1)

        super().__init__(strength=strength, num_inference_steps=num_inference_steps, **kwargs)

        self.num_inference_steps = num_inference_steps

        if isinstance(mask_image, PIL.Image.Image):
            self.mask = self.preprocess_mask(mask_image)
        else:
            self.mask = self.preprocess_mask_tensor(mask_image)

        # check sizes TODO: init_latents isn't stored or available - how to check?
        #if not self.mask.shape == self.init_latents.shape:
        #    raise ValueError("The mask and init_image should be the same size!")


        self.mask = self.mask.to(device=self.device, dtype=self.latents_dtype)
        self.mask = torch.cat([self.mask] * self.batch_total)

        # Create a mask which is either 1 (for any pixels that aren't pure black) or 0 (for pure black)
        self.high_mask = (self.mask * 100000).clamp(0, 1).round()
        # Create a mask which is either 1 (or any pixels that are pure white) or 0 (for any pixels that aren't pure white)
        self.low_mask = 1-((1-self.mask)*100000).clamp(0, 1).round()
        # Create a mask which is scaled to allow protected-area depending on how close mask_scale is to 0
        self.blend_mask = self.mask * self.mask_scale

    def _matchToSamplerSD(self, tensor):
        # Normalise tensor to -1..1
        tensor=tensor-tensor.min()
        tensor=tensor.div(tensor.max())
        tensor=tensor*2-1

        # Caculate standard deviation
        sd = tensor.std()

        if isinstance(self.scheduler, OldSchedulerMixin): 
            targetSD = self.scheduler.sigmas[0]
        else:
            targetSD = self.scheduler.init_noise_sigma

        return tensor * targetSD / sd

    def _matchNorm(self, tensor, like, cf=1):
        # Normalise tensor to 0..1
        tensor=tensor-tensor.min()
        tensor=tensor.div(tensor.max())

        # Then match range to like
        norm_range = (like.max() - like.min()) * cf
        norm_min = like.min() * cf
        return tensor * norm_range + norm_min


    def _fillWithShapedNoise(self, init_latents):
        # HERE ARE ALL THE THINGS THAT GIVE BETTER OR WORSE RESULTS DEPENDING ON THE IMAGE:
        noise_mask_factor=1 # (1) How much to reduce noise during mask transition
        lmask_mode=3 # 3 (high_mask) seems consistently good. Options are 0 = none, 1 = low mask, 2 = mask as passed, 3 = high mask
        nmask_mode=0 # 1 or 3 seem good, 3 gives good blends slightly more often
        fft_norm_mode="ortho" # forward, backward or ortho. Doesn't seem to affect results too much

        # 0 == normal, matched to latent, 1 == cauchy, matched to latent, 2 == log_normal, 3 == standard normal, mean=0, std=1
        # 0 sometimes gives the best result, but sometimes it gives artifacts
        noise_mode=0

        # 0 == to sampler requested std deviation, 1 == to original image distribution
        match_mode=1

        # Current theory: if we can match the noise to the image latents, we get a nice well scaled color blend between the two.
        # The nmask mostly adjusts for incorrect scale. With correct scale, nmask hurts more than it helps

        # noise_mode = 0 matches well with nmask_mode = 0
        # nmask_mode = 1 or 3 matches well with noise_mode = 1 or 3

        # Only consider the portion of the init image that aren't completely masked
        masked_latents = init_latents

        if lmask_mode > 0:
            latent_mask = self.low_mask if lmask_mode == 1 else self.mask if lmask_mode == 2 else self.high_mask
            masked_latents = masked_latents * latent_mask

        # Generate some noise TODO: This might affect the seed?
        noise = torch.empty_like(masked_latents)
        if noise_mode == 0 and noise_mode < 1: noise = noise.normal_(generator=self.generator, mean=masked_latents.mean(), std=masked_latents.std())
        elif noise_mode == 1 and noise_mode < 2: noise = noise.cauchy_(generator=self.generator, median=masked_latents.median(), sigma=masked_latents.std())
        elif noise_mode == 2: 
            noise = noise.log_normal_(generator=self.generator)
            noise = noise - noise.mean()
        elif noise_mode == 3: noise = noise.normal_(generator=self.generator)
        elif noise_mode == 4: 
            if isinstance(self.scheduler, OldSchedulerMixin): 
                targetSD = self.scheduler.sigmas[0]
            else:
                targetSD = self.scheduler.init_noise_sigma

            noise = noise.normal_(generator=self.generator, mean=0, std=targetSD)

        # Make the noise less of a component of the convolution compared to the latent in the unmasked portion
        if nmask_mode > 0:
            noise_mask = self.low_mask if nmask_mode == 1 else self.mask if nmask_mode == 2 else self.high_mask
            noise = noise.mul(1-(noise_mask * noise_mask_factor))

        # Color the noise by the latent
        noise_fft = torch.fft.fftn(noise.to(torch.float32), norm=fft_norm_mode)
        latent_fft = torch.fft.fftn(masked_latents.to(torch.float32), norm=fft_norm_mode)
        convolve = noise_fft.mul(latent_fft)
        noise = torch.fft.ifftn(convolve, norm=fft_norm_mode).real.to(self.latents_dtype)

        # Stretch colored noise to match the image latent
        if match_mode == 0: noise = self._matchToSamplerSD(noise)
        else: noise = self._matchNorm(noise, masked_latents, cf=1)

        # And mix resulting noise into the black areas of the mask
        return (init_latents * self.mask) + (noise * (1 - self.mask))

    def generateLatents(self):
        # Build initial latents from init_image the same as for img2img
        init_latents = self._buildInitialLatents()
        # Save the original latents for re-application in latentStep, but only the portions that definitely have pixel data
        self.init_latents_orig = init_latents * self.high_mask
        # If strength was >=1, filled exposed areas in mask with new, shaped noise
        if self.fill_with_shaped_noise: init_latents = self._fillWithShapedNoise(init_latents)
        # Add the initial noise
        init_latents = self._addInitialNoise(init_latents)
        # And return
        return init_latents

    def latentStep(self, latents, i, t, steppos):
        # The type shifting here is due to note in Img2img._addInitialNoise
        init_latents_proper = self.scheduler.add_noise(self.init_latents_orig.to(self.image_noise.dtype), self.image_noise, self._getSchedulerNoiseTimestep(i, t))       
        init_latents_proper = init_latents_proper.to(latents.dtype)

        iteration_mask = self.blend_mask.gt(steppos).to(self.blend_mask.dtype)

        return (init_latents_proper * iteration_mask) + (latents * (1 - iteration_mask))       

class DynamicModuleDiffusionPipeline(DiffusionPipeline):

    def __init__(self, *args, **kwargs):
        self._moduleMode = "all"
        self._moduleDevice = torch.device("cpu")

    def register_modules(self, **kwargs):
        self._modules = set(kwargs.keys())
        self._modulesDyn = set(("vae", "text_encoder", "unet", "safety_checker"))
        self._modulesStat = self._modules - self._modulesDyn

        super().register_modules(**kwargs)

    def set_module_mode(self, mode):
        self._moduleMode = mode
        self.to(self._moduleDevice)

    def to(self, torch_device, forceAll=False):
        if torch_device is None:
            return self

        module_names, _ = self.extract_init_dict(dict(self.config))

        self._moduleDevice = torch.device(torch_device)

        moveNow = self._modules if (self._moduleMode == "all" or forceAll) else self._modulesStat

        for name in moveNow:
            module = getattr(self, f"_{name}" if name in self._modulesDyn else name)
            if isinstance(module, torch.nn.Module):
                module.to(torch_device)
        
        return self

    @property
    def device(self) -> torch.device:
        return self._moduleDevice

    def prepmodule(self, name, module):
        if self._moduleMode == "all":
            return module

        # We assume if this module is on a device of the right type we put it there
        # (How else would it get there?)
        if self._moduleDevice.type == module.device.type:
            return module

        if name in self._modulesStat:
            return module

        for name in self._modulesDyn:
            other = getattr(self, f"_{name}")
            if other is not module: other.to("cpu")
        
        module.to(self._moduleDevice)
        return module

    @property 
    def vae(self):
        return self.prepmodule("vae", self._vae)
    
    @vae.setter 
    def vae(self, value):
        self._vae = value

    @property 
    def text_encoder(self):
        return self.prepmodule("text_encoder", self._text_encoder)
    
    @text_encoder.setter 
    def text_encoder(self, value):
        self._text_encoder = value

    @property 
    def unet(self):
        return self.prepmodule("unet", self._unet)
    
    @unet.setter 
    def unet(self, value):
        self._unet = value

    @property 
    def safety_checker(self):
        return self.prepmodule("safety_checker", self._safety_checker)
    
    @safety_checker.setter 
    def safety_checker(self, value):
        self._safety_checker = value


class NoisePredictor:

    def __init__(self, pipeline, text_embeddings, do_classifier_free_guidance, guidance_scale):
        self.pipeline = pipeline
        self.text_embeddings = text_embeddings
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale

    def step(self, latents, i, t, sigma = None):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

        if isinstance(self.pipeline.scheduler, OldSchedulerMixin): 
            if not sigma: sigma = self.pipeline.scheduler.sigmas[i] 
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
        else:
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

class UnifiedPipeline(DynamicModuleDiffusionPipeline):
    r"""
    Pipeline for unified image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """


    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()

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
        
        """
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        """
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            #safety_checker=safety_checker,       # sorry, we need all the memory we can get
            #feature_extractor=feature_extractor, # budget cuts and all that
        )
        

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        init_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        outmask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 0.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        run_safety_checker: bool = False,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if (mask_image != None and init_image == None):
            raise ValueError(f"Can't pass a mask without an image")

        if (outmask_image != None and init_image == None):
            raise ValueError(f"Can't pass a outmask without an image")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    " {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(batch_size * num_images_per_prompt, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Calculate operating mode based on arguments
        latents_dtype = text_embeddings.dtype
        batch_total = batch_size * num_images_per_prompt


        if mask_image != None: mode_class = EnhancedInpaintMode
        elif init_image != None: mode_class = Img2imgMode
        else: mode_class = Txt2imgMode

        mode = mode_class(
            pipeline=self, 
            generator=generator,
            width=width, height=height,
            init_image=init_image, mask_image=mask_image,
            latents_dtype=latents_dtype,
            batch_total=batch_total,
            num_inference_steps=num_inference_steps,
            strength=strength
        ) 

        print(f"Mode {mode.__class__} with strength {strength}")

        # Build the noise predictor. We move this into it's own class so it can be
        # passed into a scheduler if they need to re-call
        noise_predictor = NoisePredictor(
            pipeline=self, 
            text_embeddings=text_embeddings, 
            do_classifier_free_guidance=do_classifier_free_guidance, guidance_scale=guidance_scale
        )

        # Get the initial starting point - either pure random noise, or the source image with some noise depending on mode
        latents = mode.generateLatents()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        accepts_noise_predictor = "noise_predictor" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_step_kwargs = {}
        if accepts_eta: extra_step_kwargs["eta"] = eta
        if accepts_generator: extra_step_kwargs["generator"] = generator
        if accepts_noise_predictor: extra_step_kwargs["noise_predictor"] = noise_predictor.step

        t_start = mode.t_start

        timesteps_tensor = self.scheduler.timesteps[t_start:].to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            t_index = t_start + i

            # predict the noise residual
            noise_pred = noise_predictor.step(latents, t_index, t)

            # compute the previous noisy sample x_t -> x_t-1

            if isinstance(self.scheduler, OldSchedulerMixin): 
                latents = self.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            latents = mode.latentStep(latents, t_index, t, i / (timesteps_tensor.shape[0] + 1))

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        if strength <= 1 and outmask_image != None:
            outmask = torch.cat([outmask_image] * batch_size)
            outmask = outmask[:, [0,1,2]]
            outmask = outmask.to(self.device)

            source =  torch.cat([init_image] * batch_size)
            source = source[:, [0,1,2]]
            source = source.to(self.device)

            image = source * (1-outmask) + image * outmask

        numpyImage = image.cpu().permute(0, 2, 3, 1).numpy()

        if run_safety_checker:
            # run safety checker
            safety_cheker_input = self.feature_extractor(self.numpy_to_pil(numpyImage), return_tensors="pt").to(self.device)
            numpyImage, has_nsfw_concept = self.safety_checker(images=numpyImage, clip_input=safety_cheker_input.pixel_values.to(text_embeddings.dtype))
        else:
            has_nsfw_concept = [False] * numpyImage.shape[0]

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = torch.from_numpy(numpyImage).permute(0, 3, 1, 2)
        else:
            image = numpyImage

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
