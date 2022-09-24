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


g_diffuser_lib.py - core diffuser operations and lib utilities

"""

import ntpath # these lines are inexplicably required for python to use long file paths on Windows -_-
ntpath.realpath = ntpath.abspath

from g_diffuser_config import DEFAULT_PATHS, MODEL_DEFAULTS
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS

import time
import datetime
import argparse
import uuid
import pathlib
import json
import re

import numpy as np
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import skimage
from skimage.exposure import match_histograms
from skimage import color
from skimage import transform

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
    return dict(d1.copy(), **d2)
    
def valid_resolution(width, height, init_image=None):  # clip dimensions at max resolution, while keeping the correct resolution granularity,
                                                       # while roughly preserving aspect ratio. if width or height are None they are taken from the init_image
    global DEFAULT_SAMPLE_SETTINGS
    
    if not init_image:
        if not width: width = DEFAULT_SAMPLE_SETTINGS.resolution[0]
        if not height: height = DEFAULT_SAMPLE_SETTINGS.resolution[1]
    else:
        if not width: width = init_image.size[0]
        if not height: height = init_image.size[1]
        
    aspect_ratio = width / height 
    if width > DEFAULT_SAMPLE_SETTINGS.max_resolution[0]:
        width = DEFAULT_SAMPLE_SETTINGS.max_resolution[0]
        height = int(width / aspect_ratio + .5)
    if height > DEFAULT_SAMPLE_SETTINGS.max_resolution[1]:
        height = DEFAULT_SAMPLE_SETTINGS.max_resolution[1]
        width = int(height * aspect_ratio + .5)
        
    width = int(width / float(DEFAULT_SAMPLE_SETTINGS.resolution_granularity) + 0.5) * DEFAULT_SAMPLE_SETTINGS.resolution_granularity
    height = int(height / float(DEFAULT_SAMPLE_SETTINGS.resolution_granularity) + 0.5) * DEFAULT_SAMPLE_SETTINGS.resolution_granularity
    width = np.maximum(width, DEFAULT_SAMPLE_SETTINGS.resolution_granularity)
    height = np.maximum(height, DEFAULT_SAMPLE_SETTINGS.resolution_granularity)

    return int(width), int(height)
    
def get_random_string(digits=8):
    uuid_str = str(uuid.uuid4())
    return uuid_str[0:digits] # shorten uuid, don't need that many digits

def get_formatted_arg_str(args):
    namespace_dict = vars(strip_args(args))
    return json.dumps(namespace_dict, indent=4)

def print_namespace(namespace, debug=False, indent=4):
    namespace_dict = vars(strip_args(namespace))
    if debug:
        for arg in namespace_dict: print(arg+"='"+str(namespace_dict[arg]) + "' "+str(type(namespace_dict[arg])))
    else:
        print(json.dumps(namespace_dict, indent=indent))
    return
    
def get_filename_from_prompt(prompt, truncate_length=75):
    file_str = re.sub(r'[\\/*?:"<>|]',"", prompt).replace("\t"," ").replace(" ","_").strip()
    if (truncate_length > len(file_str)) or (truncate_length==0): truncate_length = len(file_str)
    return file_str[0:truncate_length]
    
def save_debug_img(np_image, name):
    global DEFAULT_PATHS
    if not DEFAULT_PATHS.debug: return
    pathlib.Path(DEFAULT_PATHS.debug).mkdir(exist_ok=True)
    
    image_path = DEFAULT_PATHS.debug + "/" + name + ".png"
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
    return image_path

def save_json(_dict, file_path):
    assert(file_path)
    (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(_dict, file, indent=4)
        file.close()
    return file_path
    
def load_json(name):
    global DEFAULT_PATHS
    assert(DEFAULT_PATHS.inputs)
    pathlib.Path(DEFAULT_PATHS.inputs).mkdir(exist_ok=True)
    
    saved_json_file_path = (pathlib.Path(DEFAULT_PATHS.inputs) / (name + ".json")).as_posix()
    with open(saved_json_file_path, "r") as file:
        data = json.load(file)
        file.close()
    return data
    
def strip_args(args, level=0): # remove args we wouldn't want to print or serialize, higher levels strip additional irrelevant fields
    args_stripped = argparse.Namespace(**(vars(args).copy()))
    if "loaded_pipes" in args_stripped: del args_stripped.loaded_pipes
    return args_stripped
    
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

def hsv_blend_image(image, match_to, hsv_mask=None):
    width = image.shape[0]
    height = image.shape[1]
    if type(hsv_mask) != np.ndarray:
        hsv_mask = np.ones((width, height, 3))
        
    image_hsv = color.rgb2hsv(image)
    match_to_hsv = color.rgb2hsv(match_to)
    return color.hsv2rgb(image_hsv * (1.-hsv_mask) + hsv_mask * match_to_hsv)
    
# prepare masks for in/out-painting
def get_blend_mask(np_mask_rgb, args):  # np_mask_rgb is an np array of rgb data in (0..1)
                                                                 # mask_blend_factor ( > 0.) adjusts blend hardness, with 1. corresponding closely to the original mask and higher values approaching the hard edge of the original mask
                                                                 # strength overrides (if > 0.) the maximum opacity in the user mask to support style transfer type applications
    assert(np_mask_rgb.ndim == 3) # needs to be a 3 channel mask
    width = np_mask_rgb.shape[0]
    height = np_mask_rgb.shape[1]
    
    if args.debug: save_debug_img(np_mask_rgb, "np_mask_rgb")
    if args.strength == 0.:
        max_opacity = np.max(np_mask_rgb)
    else:
        max_opacity = np.clip(args.strength, 0., 1.)
    
    final_blend_mask = 1. - (1.-normalize_image(gaussian_blur(1.-np_mask_rgb, std=1000.))) * max_opacity
    if args.debug: save_debug_img(final_blend_mask, "final_blend_mask")
    return final_blend_mask

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


def get_matched_noise(np_init, final_blend_mask, args): 

    width = np_init.shape[0]
    height = np_init.shape[1]
    num_channels = np_init.shape[2]
    
    # todo: experiment with transforming everything to HSV space FIRST
    windowed_image = np_init * (1.-final_blend_mask) / np.max(1.-final_blend_mask)
    if args.debug: save_debug_img(windowed_image, "windowed_src_img")
    
    assert(args.noise_q > 0.)
    noise_rgb = np.exp(-1j*2*np.pi * np.random.random_sample((width, height))) * 15. # todo: instead of 25 match with stats
    noise_rgb *= np.random.random_sample((width, height)) ** (50. * args.noise_q) # todo: instead of 50 match with stats
    noise_rgb = np.real(noise_rgb)
    colorfulness = 0. # todo: we also VERY BADLY need to control contrast and BRIGHTNESS
    noise_rgb = ((noise_rgb+0.5)*colorfulness + np_img_rgb_to_grey(noise_rgb+0.5)*(1.-colorfulness))-0.5
    
    schrodinger_kernel = get_gaussian(width, height, std=1j*2345234) * noise_rgb # todo: find a good magic number
    shaped_noise_rgb = np.absolute(convolve(schrodinger_kernel, windowed_image))
    if args.debug: save_debug_img(shaped_noise_rgb, "shaped_noise_rgb")
    
    offset = 0.01 #0.005 # 0.0125 # todo: create mask offset function that can set a lower offset
    hsv_blend_mask = (1. - final_blend_mask) * np.clip(final_blend_mask-1e-20, 0., 1.)**offset
    hsv_blend_mask = normalize_image(hsv_blend_mask)
    
    #max_opacity = np.max(hsv_blend_mask)
    hsv_blend_mask = np.minimum(normalize_image(gaussian_blur(hsv_blend_mask, std=4000.)) + 1e-8, 1.)
    offset_hsv_blend_mask = np.maximum(np.absolute(np.log(hsv_blend_mask)) ** (1/2), 0.)
    offset_hsv_blend_mask -= np.min(offset_hsv_blend_mask)
    hardness = 1 # 7.5 # 1e-8 # 0.3 
    hsv_blend_mask = normalize_image(np.exp(-hardness * offset_hsv_blend_mask**2))
    #hsv_blend_mask[:,:,0] *= 1. # todo: experiment with this again
    #hsv_blend_mask[:,:,1] *= 0.05
    #hsv_blend_mask[:,:,2] *= 0.618
    #hsv_blend_mask *= 0.95
    #hsv_blend_mask = np.ones((width, height, 3))
    if args.debug: save_debug_img(hsv_blend_mask, "hsv_blend_mask")
    
    shaped_noise_rgb = hsv_blend_image(shaped_noise_rgb, np_init, hsv_blend_mask)
    if args.debug: save_debug_img(shaped_noise_rgb, "shaped_noise_post_hsv_blend")
    
    all_mask = np.ones((width, height), dtype=bool)
    ref_mask = normalize_image(np_img_rgb_to_grey(1.-final_blend_mask))
    img_mask = ref_mask <= 0.99
    ref_mask = ref_mask > 0.1
    if args.debug:
        save_debug_img(ref_mask.astype(np.float64), "histo_ref_mask")
        save_debug_img(img_mask.astype(np.float64), "histo_img_mask")
    
    # todo: experiment with these again
    """
    matched_noise_rgb = shaped_noise_rgb.copy()
    #matched_noise_rgb[img_mask,:] = skimage.exposure.match_histograms(
    matched_noise_rgb[all_mask,:] = skimage.exposure.match_histograms(
        #shaped_noise_rgb[img_mask,:], 
        shaped_noise_rgb[all_mask,:],
        np_init[ref_mask,:],
        channel_axis=1
    )
    #shaped_noise_rgb = shaped_noise_rgb*(1.-0.618) + matched_noise_rgb*0.618
    shaped_noise_rgb = shaped_noise_rgb*0.25 + matched_noise_rgb*0.75
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_post-histo-match")
    """
    
    """
    #shaped_noise_rgb[img_mask,:] = skimage.exposure.match_histograms(
    shaped_noise_rgb[all_mask,:] = skimage.exposure.match_histograms(
        shaped_noise_rgb[all_mask,:], 
        #shaped_noise_rgb[img_mask,:], 
        np_init[ref_mask,:]**.25,
        channel_axis=1
    )
    """
    
    shaped_noise_rgb = np_init * (1.-final_blend_mask) + shaped_noise_rgb * final_blend_mask
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_post-final-blend")
    
    return np.clip(shaped_noise_rgb, 0., 1.) 
    
# ************* in/out-painting code ends here *************

def load_image(args):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS
    assert(DEFAULT_PATHS.inputs)
    final_init_img_path = (pathlib.Path(DEFAULT_PATHS.inputs) / args.init_img).as_posix()
    
    # load and resize input image to multiple of 64x64
    init_image = Image.open(final_init_img_path)
    width, height = valid_resolution(args.w, args.h, init_image=init_image)
    if (width, height) != init_image.size:
        if args.debug: print("Resizing input image to (" + str(width) + ", " + str(height) + ")")
        init_image = init_image.resize((width, height), resample=PIL.Image.LANCZOS)
    args.w = width
    args.h = height
        
    if init_image.mode == "RGBA":
        # prep masks and shaped noise, note that you only need to prep masks once if you're doing multiple samples
        mask_image = init_image.split()[-1]
        init_image = init_image.convert("RGB")
        np_init = (np.asarray(init_image.convert("RGB"))/255.).astype(np.float64)
        np_mask_rgb = (np.asarray(mask_image.convert("RGB"))/255.).astype(np.float64)
        if args.debug:
            if np.min(np_mask_rgb) > 0.: print("Warning: Image mask doesn't have any fully transparent area")
            if np.max(np_mask_rgb) < 1.: print("Warning: Image mask doesn't have any opaque area")

        if args.debug: mask_start_time = datetime.datetime.now()
        final_blend_mask = get_blend_mask(np_mask_rgb, args)
        if args.debug: print("get_blend_masks time : " + str(datetime.datetime.now() - mask_start_time))
        mask_image = PIL.Image.fromarray(np.clip(final_blend_mask*255., 0., 255.).astype(np.uint8), mode="RGB")
        
        if args.debug: noised_start_time = datetime.datetime.now()
        shaped_noise = get_matched_noise(np_init, final_blend_mask, args)
        if args.debug: print("get_matched_noise time : " + str(datetime.datetime.now() - noised_start_time))
        init_image = PIL.Image.fromarray(np.clip(shaped_noise*255., 0., 255.).astype(np.uint8), mode="RGB")
    else:
        if args.strength == 0.: args.strength = DEFAULT_SAMPLE_SETTINGS.strength
        final_blend_mask = np_img_grey_to_rgb(np.ones((args.w, args.h)) * np.clip(args.strength**(0.075), 0., 1.)) # todo: find strength mapping or do a better job of seeding
        mask_image = PIL.Image.fromarray(np.clip(final_blend_mask*255., 0., 255.).astype(np.uint8), mode="RGB")
    
    if args.debug:
        if args.strength > 0.: print("Warning: Overriding mask maximum opacity with strength : " + str(args.strength))

    return init_image, mask_image
        
def get_samples(args):
    global DEFAULT_SAMPLE_SETTINGS
    global MODEL_DEFAULTS
    
    args.status = 1 # running
    if args.init_img != "":
        pipe_name = "img2img"
        strength = 0.9999 # the real "strength" will be applied to the mask by load_image
        init_image, mask_image = load_image(args)
    else:
        pipe_name = "txt2img"
        init_image = None
        mask_image = None
        strength = None
        if not args.w: args.w = DEFAULT_SAMPLE_SETTINGS.resolution[0]
        if not args.h: args.h = DEFAULT_SAMPLE_SETTINGS.resolution[1]
        
    start_time = datetime.datetime.now()
    args.start_time = str(start_time)
    args.model_name = MODEL_DEFAULTS.model_name
    args.used_pipe = pipe_name
    samples = []
    with autocast("cuda"):
        if args.debug: print("Using " + pipe_name + " pipeline...")
        pipe = args.loaded_pipes[pipe_name]
        assert(pipe)
        for n in range(args.n): # batched mode doesn't seem to accomplish much besides using more memory
            if args.status == 3: return # if command is cancelled just bail out asap
            try:
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
            except Exception as e:
                print("Error running pipeline " + pipe_name)
                sample = None
                
            if sample: samples.append(sample["sample"][0])
    
    if len(samples) == 0: args.status = -1 # error running command
    else: args.status = 2 # completed successfully
    
    end_time = datetime.datetime.now()
    args.end_time = str(end_time)
    args.elapsed_time = str(end_time-start_time)
    if args.debug: print("total sampling time : " + args.elapsed_time)
    return samples

def save_samples(samples, args):
    if len(samples) == 0: # if there are no samples to save just return (nothing) immediately
        args.output = ""
        args.output_samples = []
        return args.output_samples 
    
    global DEFAULT_PATHS
    assert(DEFAULT_PATHS.outputs)
    if not args.outputs_path: # if no outputs_path was explicitly specified use one based on the prompt
        final_outputs_path = (pathlib.Path(DEFAULT_PATHS.outputs) / get_filename_from_prompt(args.prompt)).as_posix()
    else: # otherwise use the specified outputs_path
        final_outputs_path = (pathlib.Path(DEFAULT_PATHS.outputs) / args.outputs_path).as_posix()
    pathlib.Path(final_outputs_path).mkdir(exist_ok=True)

    # combine individual samples to create main output
    if len(samples) > 1:
        grid_layout = get_grid_layout(len(samples))
        if args.debug: print("Creating grid layout - " + str(grid_layout))
        output_image = get_image_grid(samples, grid_layout)
    else: output_image = samples[0]

    output_name = get_filename_from_prompt(args.prompt) + "__" + get_random_string()
    args.output = final_outputs_path+"/"+output_name+".png"
    if not args.no_json: args.args_output = save_json(vars(strip_args(args)), final_outputs_path+"/json/"+output_name+".json")
    output_image.save(args.output)
    print("Saved " + args.output)
    
    args.output_samples = [] # if n_samples > 1, save individual samples in tmp outputs as well
    if len(samples) > 1:
        assert(False)
        for sample in samples:
            output_name = get_filename_from_prompt(args.prompt) + "__" + get_random_string()
            output_path = final_outputs_path+"/"+output_name+".png"
            if not args.no_json: args.args_output = save_json(vars(strip_args(args)), final_outputs_path+"/json/"+output_name+".json")
            sample.save(output_path)
            print("Saved " + output_path)
            args.output_samples.append(output_path)
    else:
        args.output_samples.append(args.output) # just 1 output sample
        
    return args.output_samples
    
def load_pipelines(args):
    global DEFAULT_PATHS
    global MODEL_DEFAULTS
    
    pipe_map = { "txt2img": StableDiffusionPipeline, "img2img": StableDiffusionInpaintPipeline }
    if args.interactive:
        if "pipe_list" in MODEL_DEFAULTS: pipe_list = MODEL_DEFAULTS.pipe_list
        else: pipe_list = list(pipe_map.keys())
    else:
        if args.init_img: pipe_list = ["img2img"]
        else: pipe_list = ["txt2img"]
    
    hf_token = None
    if "hf_token" in MODEL_DEFAULTS: hf_token = MODEL_DEFAULTS.hf_token
    if "hf_token" in args: hf_token = args.hf_token
    use_optimized = MODEL_DEFAULTS.use_optimized
    if "use_optimized" in args: use_optimized = args.use_optimized
    
    if args.model_name: model_name = args.model_name
    else: model_name = MODEL_DEFAULTS.model_name
    if not hf_token: final_model_name = (pathlib.Path(DEFAULT_PATHS.models) / model_name).as_posix()
    else: final_model_name = model_name
    
    args.model_name = final_model_name
    MODEL_DEFAULTS.model_name = final_model_name
    
    print("Using model: " + final_model_name)
    if use_optimized:
        torch_dtype = torch.float16 # use fp16 in optimized mode
        print("Using memory optimizations...")
    else:
        torch_dtype = None
    
    if args.debug: load_start_time = datetime.datetime.now()
    loaded_pipes = {}
    for pipe_name in pipe_list:
        print("Loading " + pipe_name + " pipeline...")
        pipe = pipe_map[pipe_name].from_pretrained(
            final_model_name, 
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
        )
        pipe = pipe.to("cuda")
        setattr(pipe, "safety_checker", dummy_checker)
        if use_optimized == True:
            pipe.enable_attention_slicing() # use attention slicing in optimized mode
            
        loaded_pipes[pipe_name] = pipe
    if args.debug: print("load pipelines time : " + str(datetime.datetime.now() - load_start_time))

    args.loaded_pipes = loaded_pipes
    args.pipe_list = pipe_list
    args.model_name = model_name
    args.hf_token = hf_token
    return args.loaded_pipes

def get_args_parser():
    global DEFAULT_SAMPLE_SETTINGS, MODEL_DEFAULTS
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the text to condition sampling on",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="sample",
        help="diffusers command to execute",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=int(np.random.randint(DEFAULT_SAMPLE_SETTINGS.auto_seed_range[0], DEFAULT_SAMPLE_SETTINGS.auto_seed_range[1])),
        help="random seed for sampling (auto-range defined in DEFAULT_SAMPLE_SETTINGS)",
    )    
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_SAMPLE_SETTINGS.steps,
        help="number of sampling steps (number of times to refine image)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.scale,
        help="guidance scale (amount of change per step)",
    )
    parser.add_argument(
        "--init-img",
        type=str,
        default="",
        help="path to the input image",
    )
    parser.add_argument(
        "--outputs_path",
        type=str,
        help="path to store output samples (relative to root outputs path)",
        default="",
    )
    parser.add_argument(
        "--noise_q",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.noise_q,
        help="augments falloff of matched noise distribution for in/out-painting (noise_q > 0), lower values mean smaller features and higher means larger features",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.,
        help="overall amount to change the input image (default value defined in DEFAULT_SAMPLE_SETTINGS)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_SAMPLE_SETTINGS.n,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        help="set output width or override width of input image",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=None,
        help="set output height or override height of input image",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_DEFAULTS.model_name,
        help="path to downloaded diffusers model (relative to default models path), or name of model if using a huggingface token",
    )
    parser.add_argument(
        "--use_optimized",
        action='store_true',
        default=False,
        help="enable memory optimizations that are currently available in diffusers",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="enable verbose CLI output and debug image dumps",
    )
    parser.add_argument(
        "--interactive",
        action='store_true',
        default=False,
        help="enters an interactive command line mode to generate multiple samples",
    )
    parser.add_argument(
        "--load-args",
        type=str,
        default="no_preload",
        help="preload and use a saved set of arguments from a json file in your inputs path",
    )
    parser.add_argument(
        "--no-json",
        action='store_true',
        default=False,
        help="disable saving arg files for each sample output in output path/json",
    )
    parser.add_argument(
        "--uuid-str",
        type=str,
        default=get_random_string(),
        help="attach an id (default random) that can be used for identification or searching purposes later",
    )
    return parser