import inspect
import warnings
import gc
from typing import List, Optional, Union

import numpy as np
import torch
import PIL
import torchvision.transforms as T

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from sdgrpcserver.pipeline.scheduling_ddim import DDIMScheduler
from sdgrpcserver.pipeline.scheduling_euler_discrete import EulerDiscreteScheduler
from sdgrpcserver.pipeline.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def preprocess_mask(mask):
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



class UnifiedPipeline(DiffusionPipeline):
    r"""
    General Stable Diffusion Pipeline. Merge of text-to-image, image-to-image and inpaint pipelines, plus some
    other modifications, in particular:

        - Is possible to move _just_ the unet to cuda and have the pipeline still run (reduces VRAM)
        - Negative prompting

        TODO:

        - Remove needing autocast for FP16 (big speedup for FP16)

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
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
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
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
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
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

    def enable_minimal_memory_usage(self):
        """Moves only unet to fp16 and to CUDA, while keepping lighter models on CPUs"""
        self.unet.to(torch.float16).to(torch.device("cuda"))
        self.enable_attention_slicing(1)

        torch.cuda.empty_cache()
        gc.collect()

    def match_norm(self, tensor, like, cf=1):
        # Normalise tensor to 0..1
        tensor=tensor-tensor.min()
        tensor=tensor.div(tensor.max())

        # Then match range to like
        norm_range = (like.max() - like.min()) * cf
        norm_min = like.min() * cf
        return tensor * norm_range + norm_min

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        mask_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        strength: Optional[float] = 0.8,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        When init_image is not provided, runs a text to image process
        When init_image is provided, but a mask isn't, runs an image to image process
        When init_image and mask is provided, runs an inpaint process

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
               `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. The mask image will be
                converted to a single channel (luminance) before use.
            strength (`float`, *optional*, defaults to 0.8):
                Ignored unless init_image is provided.
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.            
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
                The prompt or prompts not to guide the image generation.
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
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

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

        # Calculate operating mode based on arguments
        if mask_image: mode = "inpaint"
        elif init_image: mode = "img2img"
        else: mode = "txt2img"

        max_strength = 2.0 if mode == "inpaint" else 1.0

        if mode == "txt2img":
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        elif strength < 0 or strength > max_strength:
            raise ValueError(f"The value of strength should in [0.0, {max_strength}] but is {strength}")

        print(f"Mode {mode} with strength {strength}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        if mode == "txt2img":
            # get the initial random noise unless the user supplied it

            # Unlike in other pipelines, latents need to be generated in the target device
            # for 1-to-1 results reproducibility with the CompVis implementation.
            # However this currently doesn't work in `mps`.
            latents_device = "cpu" if self.device.type == "mps" else self.device
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
            if latents is None:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=latents_device,
                )
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
                latents = latents * self.scheduler.sigmas[0]

            t_start = 0

        else:
            if isinstance(init_image, PIL.Image.Image):
                init_image = preprocess(init_image)

            if mode == "inpaint": init_image = init_image.to(self.device)

            # encode the init image into latents and scale the latents
            init_latent_dist = self.vae.encode(init_image.to(self.device)).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * batch_size)

            if mode == "inpaint":
                init_latents_orig = init_latents

                # preprocess mask
                if isinstance(mask_image, PIL.Image.Image):
                    mask = preprocess_mask(mask_image)
                else:
                    mask = mask_image

                mask = mask.to(self.device)
                mask = torch.cat([mask] * batch_size)

                # Mask is now "white keep, black discard" (preprocess_mask inverts image)

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")

                if strength >= 1:
                    # HERE ARE ALL THE THINGS THAT GIVE BETTER OR WORSE RESULTS DEPENDING ON THE IMAGE:
                    noise_mask_factor=1 # (1) How much to reduce noise during mask transition
                    lmask_mode=3 # 3 (high_mask) seems consistently good. Options are 0 = none, 1 = low mask, 2 = mask as passed, 3 = high mask
                    nmask_mode=0 # 1 or 3 seem good, 3 gives good blends slightly more often
                    fft_norm_mode="ortho" # forward, backward or ortho. Doesn't seem to affect results too much

                    # 0 == normal, matched to latent, 1 == cauchy, matched to latent, 2 == log_normal, 3 == standard normal, mean=0, std=1
                    # 0 sometimes gives the best result, but sometimes it gives artifacts
                    noise_mode=0

                    mask_scale=2-strength # How much to scale the mask down by (limits mask to allow changes even in protected area)

                    # Current theory: if we can match the noise to the image latents, we get a nice well scaled color blend between the two.
                    # The nmask mostly adjusts for incorrect scale. With correct scale, nmask hurts more than it helps

                    # noise_mode = 0 matches well with nmask_mode = 0
                    # nmask_mode = 1 or 3 matches well with noise_mode = 1 or 3

                    # Reset strength to just be 1 (we currently overload it so "component over 1" can be used to control a parameter above during testing)
                    strength=1

                    # Create a mask which is either 1 (for any pixels that aren't pure black) or 0 (for pure black)
                    high_mask = (mask * 100000).clamp(0, 1).round()
                    # Create a mask which is either 1 (or any pixels that are pure white) or 0 (for any pixels that aren't pure white)
                    low_mask = 1-((1-mask)*100000).clamp(0, 1).round()

                    # Only consider the portion of the init image that aren't completely masked
                    masked_latents = init_latents

                    if lmask_mode > 0:
                        latent_mask = low_mask if lmask_mode == 1 else mask if lmask_mode == 2 else high_mask
                        masked_latents = masked_latents * latent_mask

                    # Generate some noise TODO: This might affect the seed?
                    noise = torch.empty_like(masked_latents)
                    if noise_mode == 0 and noise_mode < 1: noise = noise.normal_(generator=generator, mean=masked_latents.mean(), std=masked_latents.std())
                    elif noise_mode == 1 and noise_mode < 2: noise = noise.cauchy_(generator=generator, median=masked_latents.median(), sigma=masked_latents.std())
                    elif noise_mode == 2: 
                        noise = noise.log_normal_(generator=generator)
                        noise = noise - noise.mean()
                    elif noise_mode == 3: noise = noise.normal_(generator=generator)

                    # Make the noise less of a component of the convolution compared to the latent in the unmasked portion
                    if nmask_mode > 0:
                        noise_mask = low_mask if nmask_mode == 1 else mask if nmask_mode == 2 else high_mask
                        noise = noise.mul(1-(noise_mask * noise_mask_factor))

                    # Color the noise by the latent
                    noise_fft = torch.fft.fftn(noise, norm=fft_norm_mode)
                    latent_fft = torch.fft.fftn(masked_latents, norm=fft_norm_mode)
                    convolve = noise_fft.mul(latent_fft)
                    noise = torch.fft.ifftn(convolve, norm=fft_norm_mode).real

                    # Stretch colored noise to match the image latent
                    noise = self.match_norm(noise, masked_latents, cf=1)

                    # And mix resulting noise into the black areas of the mask
                    init_latents = (init_latents_orig * mask) + (noise * (1 - mask))

                    # For EulerA, we get a nicer result when we clip mask below 1 max
                    # BUT (TODO: check) only when masked portions are truely black. So enforce that on orig.
                    init_latents_orig = init_latents_orig * high_mask
                    mask = mask * mask_scale
                    #mask = mask.clamp(0, mask_clip_max)

            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
                timesteps = torch.tensor(
                    [num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=self.device
                )
            else:
                timesteps = self.scheduler.timesteps[-init_timestep]
                timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

            # add noise to latents using the timesteps
            noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps).to(self.device)

            latents = init_latents
            t_start = max(num_inference_steps - init_timestep + offset, 0)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0].to(self.unet.device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            ucond_tokens: List[str]
            if negative_prompt is None:
                ucond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError("`negative_prompt` should be the same type to `prompt`.")
            elif isinstance(negative_prompt, str):
                ucond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                ucond_tokens = negative_prompt

            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                ucond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0].to(self.unet.device)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_step_kwargs = {}
        if accepts_eta: extra_step_kwargs["eta"] = eta
        if accepts_generator: extra_step_kwargs["generator"] = generator

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[t_start:])):
            t_index = t_start + i

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):            
                sigma = self.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                latent_model_input = latent_model_input.to(self.unet.dtype)
                t = t.to(self.unet.dtype)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input.to(self.unet.device), t.to(self.unet.device), encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if mode == "inpaint":
                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
                    latents = self.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs).prev_sample
                    # masking
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor(t_index))
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    # masking
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)

                # Calculate mask for this iteration
                # We want 1 to merge original, 0 to merge current latent
                # For each mask cell, we want original if current step position (0..1) is less than mask value
                # TODO: Not sure if this calculation is reliable for strength < 1.
                steppos = i / num_inference_steps
                iteration_mask = mask.ge(steppos).to(mask.dtype)

                latents = (init_latents_proper * iteration_mask) + (latents * (1 - iteration_mask))
            else:
                if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
                    latents = self.scheduler.step(noise_pred, t_index, latents.to(self.unet.device), **extra_step_kwargs).prev_sample
                else:
                    latents = self.scheduler.step(noise_pred, t.to(self.unet.device), latents.to(self.unet.device), **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents.to(self.vae.device)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.to(self.vae.device).cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
