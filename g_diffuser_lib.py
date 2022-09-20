"""
MIT License

Copyright (c) 2022 Christopher Friesen
https://github.com/parlance-zz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


g_diffuser_lib.py - shared functions and diffusers operations

"""

from g_diffuser_bot_defaults import *

import time
import datetime
import argparse
import uuid
import pathlib

import numpy as np
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import skimage
from skimage.exposure import match_histograms
from skimage import color

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline # we don't need the img2img pipeline because inpaint is a superset of its functionality
#from diffusers import LMSDiscreteScheduler          # broken at the moment I believe

def get_image_dims(img_path):
    img = Image.open(img_path)
    size = img.size
    img.close()
    return size

def merge_dicts(d1, d2): # overwrites the attributes in d1 in merge
    return dict(d1, **d2)
    
def valid_resolution(width, height, init_image=None): # cap max dimension at max res and ensure size is 
                                                       # a correct multiple of granularity while
                                                       # preserving aspect ratio (best we can anyway)
    global DEFAULT_RESOLUTION
    global MAX_RESOLUTION
    
    RESOLUTION_GRANULARITY = 64 # hard-coded for SD for now
    
    if not init_image:
        if not width: width = DEFAULT_RESOLUTION[0]
        if not height: height = DEFAULT_RESOLUTION[1]
    else:
        if not width: width = init_image.size[0]
        if not height: height = init_image.size[1]
        
    aspect_ratio = width / height 
    if width > MAX_RESOLUTION[0]:
        width = MAX_RESOLUTION[0]
        height = int(width / aspect_ratio + .5)
    if height > MAX_RESOLUTION[1]:
        height = MAX_RESOLUTION[1]
        width = int(height * aspect_ratio + .5)
        
    width = int(width / float(RESOLUTION_GRANULARITY) + 0.5) * RESOLUTION_GRANULARITY
    height = int(height / float(RESOLUTION_GRANULARITY) + 0.5) * RESOLUTION_GRANULARITY
    if width < RESOLUTION_GRANULARITY: width = RESOLUTION_GRANULARITY
    if height < RESOLUTION_GRANULARITY: height = RESOLUTION_GRANULARITY

    return width, height
    
def get_tmp_path(file_extension):
    global TMP_ROOT_PATH
    try: # try to make sure temp folder exists
        pathlib.Path(TMP_ROOT_PATH).mkdir(exist_ok=True)
    except Exception as e:
        print("Error creating temp path: '" + TMP_ROOT_PATH + "' - " + str(e))
        
    uuid_str = str(uuid.uuid4())
    uuid_str = uuid_str[0:len(uuid_str)//2] # shorten uuid, don't need that many digits
    tmp_path = pathlib.Path(TMP_ROOT_PATH) / (uuid_str + file_extension)
    return tmp_path.absolute().as_posix()

def save_debug_img(np_image, name):
    global DEBUG_MODE
    if not DEBUG_MODE: return
    global TMP_ROOT_PATH
    if not TMP_ROOT_PATH: return
    
    try: # try to make sure temp folder exists
        pathlib.Path(TMP_ROOT_PATH).mkdir(exist_ok=True)
    except:
        return
    
    image_path = TMP_ROOT_PATH + "/_debug_" + name + ".png"
    if type(np_image) == np.ndarray:
        if np_image.ndim == 2:
            mode = "L"
        elif np_image.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        pil_image = PIL.Image.fromarray(np.clip(np.absolute(np_image)*255., 0., 255.).astype(np.uint8), mode=mode)
        pil_image.save(image_path)
    else:
        np_image.save(image_path)
    
def dummy_checker(images, **kwargs): # replacement func to disable safety_checker in diffusers
    return images, False

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]
    
def get_grid_layout(num_samples):
    factors = factorize(num_samples)
    median_factor = factors[len(factors)//2]
    columns = median_factor
    rows = num_samples // columns
    
    return (rows, columns)
    
def get_image_grid(imgs, layout): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(layout[1]*w, layout[0]*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%layout[1]*w, i//layout[1]*h))
    return grid


# ************* in/out-painting code begins here *************

# helper fft routines that keep ortho normalization and auto-shift before and after fft, and can handle multi-channel images

def fft2(data):
    if data.ndim > 2: # multiple channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # single channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft
   
def ifft2(data):
    if data.ndim > 2: # multiple channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # single channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft
            
def get_gaussian(width, height, std=3.14, edge_filter=False): # simple gaussian kernel

    window_scale_x = float(width / min(width, height))  # for non-square aspect ratios we still want a circular gaussian
    window_scale_y = float(height / min(width, height)) 
    window = np.zeros((width, height))
    
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    kx = np.exp(-x*x * std)
    if window_scale_x != window_scale_y:
        y = (np.arange(height) / height * 2. - 1.) * window_scale_y
        ky = np.exp(-y*y * std)
    else:
        y = x
        ky = kx
    gaussian = np.outer(kx, ky)
    
    if edge_filter:
        return gaussian * (1. -std*np.add.outer(x*x,y*y)) # normalized gaussian 2nd derivative
    else:
        return gaussian

def convolve(data1, data2):      # fast convolution with fft
    if data1.ndim != data2.ndim: # promote to rgb if mismatch
        if data1.ndim < 3: data1 = np_img_grey_to_rgb(data1)
        if data2.ndim < 3: data2 = np_img_grey_to_rgb(data2)
    return ifft2(fft2(data1) * fft2(data2))

def gaussian_blur(data, std=3.14):
    width = data.shape[0]
    height = data.shape[1]
    kernel = get_gaussian(width, height, std)
    return np.real(convolve(data, kernel / np.sqrt(np.sum(kernel*kernel))))
 
def normalize_image(data):
    normalized = data - np.min(data)
    normalized_max = np.max(normalized)
    assert(normalized_max > 0.)
    return normalized / normalized_max
 
def np_img_rgb_to_grey(data):
    if data.ndim == 2: return data
    return np.sum(data, axis=2)/3.
    
def np_img_grey_to_rgb(data):
    if data.ndim == 3: return data
    return np.expand_dims(data, 2) * np.ones((1, 1, 3))

def hsv_blend_image(image, match_to, hsv_mask):
    image_hsv = color.rgb2hsv(image)
    match_to_hsv = color.rgb2hsv(match_to)
    return color.hsv2rgb(image_hsv * (1.-hsv_mask) + hsv_mask * match_to_hsv)
    
# prepare masks for in/out-painting
def get_blend_masks(np_mask_rgb, mask_blend_factor, strength):  # np_mask_rgb is an np array of rgb data in (0..1)
                                                                 # mask_blend_factor ( > 0.) adjusts blend hardness, with 1. corresponding closely to the original mask and higher values approaching the hard edge of the original mask
                                                                 # strength overrides (if > 0.) the maximum opacity in the user mask to support style transfer type applications
    assert(np_mask_rgb.ndim == 3) # needs to be a 3 channel mask
    width = np_mask_rgb.shape[0]
    height = np_mask_rgb.shape[1]
    
    assert(mask_blend_factor > 0.)
    assert(np.average(np_mask_rgb) > 0.) # images where every pixel is fully masked will trigger this assert
    save_debug_img(np_mask_rgb, "np_mask_rgb")
    
    mask_hardened = 1. - (np_mask_rgb > 0.).astype(np.float64)
    save_debug_img(mask_hardened, "mask_hardened")

    # this fits a blurred interior mask that avoids bleeding over the outside of the original mask
    blend_mask = np.clip(gaussian_blur(mask_hardened*1250.,std=500.), 0., 1.)
    blend_mask /= np.max(blend_mask)
    save_debug_img(blend_mask, "blend_mask")
    
    final_blend_mask = blend_mask ** mask_blend_factor             # and adjust hardness by the blend factor
    final_blend_mask = gaussian_blur(final_blend_mask*1.1, std=500.)  # one final very small blur to prevent aliasing from blend factor or bad user masks
    final_blend_mask = normalize_image(final_blend_mask)
     
    window_mask = np.minimum(final_blend_mask[:] + 0.025, 1.)
    assert(np.average(window_mask) < 1.)
    save_debug_img(window_mask, "window_mask")
    
    if strength == 0.:
        max_opacity = np.max(np_mask_rgb)
    else:
        max_opacity = np.clip(strength, 0., 1.)
    
    final_blend_mask = 1. - (1.-final_blend_mask) *  max_opacity # scale the blend mask by the max opacity of the original mask to support partial blending / style transfer
    save_debug_img(final_blend_mask, "final_blend_mask")
    return mask_hardened, final_blend_mask, window_mask


"""

 Why does this need to exist? I thought SD already did in/out-painting?:
 
 This seems to be a common misconception. Non-latent diffusion models such as Dall-e can be readily used for in/out-painting
 but the current SD in-painting pipeline is just regular img2img with a mask, and changing that would require training a
 completely new model (at least to my understanding). In order to get good results, SD needs to have information in the
 (completely) erased area of the image. Adding to the confusion is that the PNG file format is capable of saving color data in
 (completely) erased areas of the image but most applications won't do this by default, and copying the image data to the "clipboard"
 will erase the color data in the erased regions (at least in Windows). Code like this or patchmatch that can generate a
 seed image (or "fixed code") will (at least for now) be required for seamless out-painting.
 
 Although there are simple effective solutions for in-painting, out-painting can be especially challenging because there is no color data
 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.

 By taking a fourier transform of the unmasked source image we get a function that tells us the presence, orientation, and scale of features
 in that source. Shaping the init/seed/fixed code noise to the same distribution of feature scales, orientations, and positions/phases
 increases (visual) output coherence by helping keep features aligned and of similar orientation and size. This technique is applicable to any continuous
 generation task such as audio or video, each of which can be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased.
 TLDR: The fourier transform of the unmasked source image is a strong prior for shaping the noise distribution of in/out-painted areas
 
 For multi-channel data such as color or stereo sound the "color tone" of the noise can be bled into the noise with gaussian convolution and
 a final histogram match to the unmasked source image ensures the palette of the source is mostly preserved. SD is extremely sensitive to
 careful color and "texture" matching to ensure features are appropriately "bound" if they neighbor each other in the transition zone.
 
 The effects of both of these techiques in combination include helping the generator integrate the pre-existing view distance and camera angle,
 as well as being more likely to complete partially erased features (like appropriately completing a partially erased arm, house, or tree).
 
 Please note this implementation is written for clarity and correctness rather than performance.
 
 Todo: To be investigated is the idea of using the same technique directly in latent space. Spatial properties are (at least roughly?) preserved
 in latent space so the fourier transform should be usable there for the same reason convolutions are usable there. The ideas presented here
 could also be combined or augmented with other existing techniques.
 Todo: It would be trivial to add brightness, contrast, and overall palette control using simple parameters
 Todo: There are some simple optimizations that can increase speed significantly, e.g. re-using FFTs and gaussian kernels
 
 Parameters:
 
 - np_init should be an np array of the RGB source image in range (0..1)
 - noise_q modulates the fall-off of the target distribution and can be any positive number, lower means higher detail in the in/out-painted area (range > 0, default 1., good values are usually near 1.5)
 - mask_hardened, final_blend_mask and window_mask are pre-prepared masks for which another function is provided above. Quality is highly sensitive
   to the construction of these masks.

 Dependencies: numpy, scikit-image

 This code is provided under the MIT license -  Copyright (c) 2022 Christopher Friesen
 To anyone who reads this I am seeking employment in related areas.
 Donations would also be greatly appreciated and will be used to fund further development. (ETH @ 0x8e4BbD53bfF9C0765eE1859C590A5334722F2086)
 
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 
"""


def get_matched_noise(np_init, mask_hardened, final_blend_mask, window_mask, noise_q): 

    assert(noise_q > 0.)
    
    width = np_init.shape[0]
    height = np_init.shape[1]
    num_channels = np_init.shape[2]
    
    """
    edge_kernel = get_gaussian(width, height, std=1e5, edge_filter=True)
    edges = np.real(convolve(np_init, edge_kernel)) * (1.-window_mask) **0.01 #........................
    edges /= np.max(edges)
    save_debug_img(edges, "edges")
    
    windowed_image = np_init * (1.-window_mask)
    color_bleed = np.zeros((width, height, num_channels)) # start with just the average color of the unmasked source
    for c in range(num_channels):
        color_bleed[:,:,c] = np.clip(np.average(windowed_image[:,:,c]) / np.average(1.-window_mask[:,:,c]), 0., 1.)
    for i in range(3):
        std = 16. * (10**i)
        blurred = normalize_image(gaussian_blur(edges*2., std=std))
        color_bleed *= blurred + 1.
        
    color_bleed /= np.max(color_bleed)
    color_bleed = normalize_image(color_bleed)
    #color_bleed = normalize_image_brightness(color_bleed)
    save_debug_img(color_bleed, "color_bleed")
    
    #windowed_image += color_bleed * window_mask
    """
    windowed_image = np_init[:]
    save_debug_img(windowed_image, "windowed_src_img")
    
    
    """
    #windowed_image_grey = np_img_rgb_to_grey(windowed_image)
    #src_fft = fft2(windowed_image_grey) # get source image feature statistics
    src_fft = fft2(windowed_image) # get source image feature statistics
    
    src_dist = np.absolute(src_fft[:])
    src_dist_max = np.max(src_dist)
    assert(src_dist_max > 0.)
    src_phase = src_fft / src_dist
    src_dist = ((src_dist / src_dist_max) ** noise_q) * src_dist_max  # apply falloff from noise_q
    
    #src_dist_rgb = _np_img_grey_to_rgb(src_dist)
    #src_phase_rgb = _np_img_grey_to_rgb(src_phase)
    
    src_dist_rgb = src_dist[:]
    src_phase_rgb = src_phase[:]
    
    save_debug_img(normalize_image(src_dist_rgb)*1e5, "windowed_src_dist")
    save_debug_img(normalize_image(np.angle(src_phase_rgb)), "windowed_src_phase")
    
    #noise = np.random.random_sample((width, height))*2. - 1.
    #noise_rgb = _np_img_grey_to_rgb(noise)
    noise_rgb = np.random.random_sample((width, height, num_channels)) 
    
    shaped_noise_fft = fft2(noise_rgb)*2. - 1.
    shaped_noise_fft_abs = np.absolute(shaped_noise_fft)
    #shaped_noise_fft = shaped_noise_fft * src_dist_rgb # * src_phase_rgb #* 3.14 # perform the actual shaping
    shaped_noise_fft = np.absolute(shaped_noise_fft) * src_dist_rgb * src_phase_rgb #* 3.14 # perform the actual shaping
    shaped_noise_fft -= np.average(shaped_noise_fft)
    shaped_noise_fft /= np.sum(np.absolute(shaped_noise_fft)**2)**0.5
    shaped_noise_fft *= np.sum(np.absolute(src_fft - np.average(src_fft))**2)**0.5
    shaped_noise_fft += np.average(src_fft)
    
    save_debug_img(normalize_image(np.absolute(shaped_noise_fft))*1e5, "shaped_noise_fft_postshape")
    save_debug_img(normalize_image(np.angle(shaped_noise_fft)), "shaped_noise_phase_postshape")
    
    shaped_noise_rgb = np.real(ifft2(shaped_noise_fft))
    shaped_noise_rgb = normalize_image(shaped_noise_rgb)
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_pre-colorize")
    shaped_noise_rgb = hsv_blend_image(shaped_noise_rgb, color_bleed)
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_pre-histo-match")
    """
    
    schrodinger_kernel = get_gaussian(width, height, std=1j*width*2. + noise_q) * (np.random.random_sample((width, height))*2. - 1.)
    shaped_noise_rgb = np.absolute(convolve(schrodinger_kernel, windowed_image))
    shaped_noise_rgb = normalize_image(shaped_noise_rgb)
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb")
    
    #shaped_noise_rgb = np_init * (1. - final_blend_mask**.5) + shaped_noise_rgb * final_blend_mask**.5
    #save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_wtf2")
    
    #colorized = hsv_blend_image(shaped_noise_rgb, np_init)
    hsv_blend_mask = (1. - final_blend_mask)
    #hsv_blend_mask[:,:,0] *= 0.5
    #hsv_blend_mask[:,:,1] *= 0.7071
    #hsv_blend_mask[:,:,2] *= 0.7071
    shaped_noise_rgb = hsv_blend_image(shaped_noise_rgb, np_init, hsv_blend_mask)
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_colorized")
    
    #shaped_noise_rgb = colorized * (1. - final_blend_mask) + shaped_noise_rgb * final_blend_mask
    #save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_colorized_blend")
    
    #shaped_noise_rgb = np_init * (1. - final_blend_mask) + shaped_noise_rgb * final_blend_mask
    #save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_final_blend")

    
    """
    all_mask = np.ones((width, height), dtype=bool)
    ref_mask = np_img_rgb_to_grey(mask_hardened) < 0.99    
    shaped_noise_rgb[all_mask,:] = skimage.exposure.match_histograms(
        shaped_noise_rgb[all_mask,:], 
        np_init[ref_mask,:]**2,
        channel_axis=1
    )
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_post-histo-match")
    """
    
    #shaped_noise_rgb = np_init * (1. - final_blend_mask) + shaped_noise_rgb * final_blend_mask
    #save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_post-final-blend")
    
    return np.clip(shaped_noise_rgb, 0., 1.) 
    
# ************* in/out-painting code ends here *************

def load_image(args):
    global DEBUG_MODE
    
    # load and resize input image to multiple of 64x64
    init_image = Image.open(args.init_img)
    width, height = valid_resolution(args.w, args.h, init_image=init_image)
    if (width, height) != init_image.size:
        if DEBUG_MODE: print("Resizing input image to (" + str(width) + ", " + str(height) + ")")
        init_image = init_image.resize((width, height), resample=PIL.Image.LANCZOS)
    args.w = width
    args.h = height
        
    if init_image.mode == "RGBA":
        # prep masks and shaped noise, note that you only need to prep masks once if you're doing multiple samples
        mask_image = init_image.split()[-1]
        init_image = init_image.convert("RGB")
        np_init = (np.asarray(init_image.convert("RGB"))/255.).astype(np.float64)
        np_mask_rgb = (np.asarray(mask_image.convert("RGB"))/255.).astype(np.float64)
        if DEBUG_MODE:
            if np.min(np_mask_rgb) > 0.: print("Warning: Image mask doesn't have any fully transparent area")
            if np.max(np_mask_rgb) < 1.: print("Warning: Image mask doesn't have any opaque area")

        if DEBUG_MODE: mask_start_time = datetime.datetime.now()
        mask_hardened, final_blend_mask, window_mask = get_blend_masks(np_mask_rgb, args.blend, args.strength)
        if DEBUG_MODE: print("get_blend_masks time : " + str(datetime.datetime.now() - mask_start_time))
        mask_image = PIL.Image.fromarray(np.clip(final_blend_mask*255., 0., 255.).astype(np.uint8), mode="RGB")
        
        if DEBUG_MODE: noised_start_time = datetime.datetime.now()
        shaped_noise = get_matched_noise(np_init, mask_hardened, final_blend_mask, window_mask, args.noise_q)
        if DEBUG_MODE: print("get_matched_noise time : " + str(datetime.datetime.now() - noised_start_time))
        init_image = PIL.Image.fromarray(np.clip(shaped_noise*255., 0., 255.).astype(np.uint8), mode="RGB")
    else:
        if args.strength == 0.: args.strength = 0.5 # todo: non-hardcoded default
        final_blend_mask = np_img_grey_to_rgb(np.ones((args.w, args.h)) * np.clip(args.strength, 0., 1.))
        mask_image = PIL.Image.fromarray(np.clip(final_blend_mask*255., 0., 255.).astype(np.uint8), mode="RGB")
    
    if DEBUG_MODE:
        if args.strength > 0.: print("Warning: Overriding mask maximum opacity with strength : " + str(args.strength))

    return init_image, mask_image
        
def get_samples(args):
    global DEBUG_MODE
    global DEFAULT_RESOLUTION
    
    if args.init_img != "":
        pipe_name = "img2img"
        init_image, mask_image = load_image(args)
        strength = 0.9999 # the real "strength" will be applied to the mask by load_image
    else:
        pipe_name = "txt2img"
        init_image = None
        mask_image = None
        strength = None
        if not args.w: args.w = DEFAULT_RESOLUTION[0]
        if not args.h: args.h = DEFAULT_RESOLUTION[1]
        
    if DEBUG_MODE:
        sampling_start_time = datetime.datetime.now()
        print("Using " + pipe_name + " pipeline...")
    
    args.used_pipe = pipe_name
    samples = []
    with autocast("cuda"):
        pipe = args.loaded_pipes[pipe_name]
        assert(pipe)
        
        for n in range(args.n): # batched mode doesn't seem to accomplish much besides using more memory
            if pipe_name == "txt2img":
                sample = pipe(
                    prompt=args.prompt,
                    guidance_scale=args.scale,
                    num_inference_steps=args.steps,
                    width=args.w,
                    height=args.h,
                )
            else:
                sample = pipe(
                    prompt=args.prompt,
                    init_image=init_image,
                    strength=strength,
                    guidance_scale=args.scale,
                    mask_image=mask_image,
                    num_inference_steps=args.steps,
                )
            samples.append(sample["sample"][0])

    if DEBUG_MODE: print("total sampling time : " + str(datetime.datetime.now() - sampling_start_time))
    return samples

def save_samples(samples, args):
    # combine individual samples to create main output
    if len(samples) > 1: output_image = get_image_grid(samples, get_grid_layout(len(samples)))
    else: output_image = samples[0]

    # todo: temporarily break output path specification while we refactor to support target folders instead
    
    args.output = get_tmp_path(".png")
    output_image.save(args.output)
    print("Saved " + args.output)
    
    args.output_samples = [] # if n_samples > 1, save individual samples in tmp outputs as well
    if len(samples) > 1:
        for sample in samples:
            output_path = get_tmp_path(".png")
            sample.save(output_path)
            print("Saved " + output_path)
            args.output_samples.append(output_path)
    else:
        args.output_samples.append(args.output) # just 1 output sample
    
    # todo: save json files for each output with args
    
    return args.output_samples
    
def load_pipelines(args):
    global DEBUG_MODE
    global HUGGINGFACE_TOKEN
    global CMD_SERVER_MODEL_NAME
    global BOT_USE_OPTIMIZED
    global LOAD_PIPE_LIST
    
    pipe_map = { "txt2img": StableDiffusionPipeline, "img2img": StableDiffusionInpaintPipeline }
    if args.interactive:
        if LOAD_PIPE_LIST != None: pipe_list = LOAD_PIPE_LIST
        else: pipe_list = list(pipe_map.keys())
    else:
        if args.init_img: pipe_list = ["img2img"]
        else: pipe_list = ["txt2img"]
            
    if not args.use_optimized: use_optimized = BOT_USE_OPTIMIZED
    else: use_optimized = args.use_optimized
    if not args.model_name: args.model_name = CMD_SERVER_MODEL_NAME
    if not args.hf_token: hf_token = HUGGINGFACE_TOKEN
    
    if use_optimized:
        torch_dtype = torch.float16 # use fp16 in optimized mode
        if DEBUG_MODE: print("Using memory optimizations...")
    else:
        torch_dtype = None
    
    if DEBUG_MODE: load_start_time = datetime.datetime.now()
    loaded_pipes = {}
    for pipe_name in pipe_list:
        print("Loading " + pipe_name + " pipeline...")
        pipe = pipe_map[pipe_name].from_pretrained(
            args.model_name, 
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
        )
        pipe = pipe.to("cuda")
        setattr(pipe, "safety_checker", dummy_checker)
        if use_optimized == True:
            pipe.enable_attention_slicing() # use attention slicing in optimized mode
            
        loaded_pipes[pipe_name] = pipe
    if DEBUG_MODE: print("load pipelines time : " + str(datetime.datetime.now() - load_start_time))

    args.loaded_pipes = loaded_pipes
    args.pipe_list = pipe_list
    return args.loaded_pipes
