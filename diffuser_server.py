from g_diffuser_bot_params import *

import os, sys
os.chdir(ROOT_PATH)

import argparse
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import skimage
from skimage.exposure import match_histograms

import PIL
import numpy as np
import asyncio
import json
import pathlib
import uuid

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import LMSDiscreteScheduler

DEFAULT_RESOLUTION = (512, 512)
RESOLUTION_GRANULARITY = 64

def _valid_resolution(width, height, init_image=None): # cap max dimension at max res and ensure size is 
                                                       # a correct multiple of granularity while
                                                       # preserving aspect ratio (best we can anyway)
    global RESOLUTION_GRANULARITY
    global DEFAULT_RESOLUTION
    global MAX_RESOLUTION
    
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
    
def _get_tmp_path(file_extension):
    global TMP_ROOT_PATH
    try: # try to make sure temp folder exists
        pathlib.Path(TMP_ROOT_PATH).mkdir(exist_ok=True)
    except Exception as e:
        print("Error creating temp path: '" + TMP_ROOT_PATH + "' - " + str(e))
    return TMP_ROOT_PATH + "/" + str(uuid.uuid4()) + file_extension

def _save_debug_img(np_image, name):
    global DEBUG_MODE
    if not DEBUG_MODE: return
    global TMP_ROOT_PATH
    
    image_path = TMP_ROOT_PATH + "/_debug_" + name + ".png"
    if type(np_image) == np.ndarray:
        if np_image.ndim == 2:
            mode = "L"
        elif np_image.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        pil_image = PIL.Image.fromarray(np.clip(np_image*255., 0., 255.).astype(np.uint8), mode=mode)
        pil_image.save(image_path)
    else:
        np_image.save(image_path)
    
def _dummy_checker(images, **kwargs): # replacement func to disable safety_checker in diffusers
    return images, False

def _factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]
    
def _get_grid_layout(num_samples):
    factors = _factorize(num_samples)
    median_factor = factors[len(factors)//2]
    columns = median_factor
    rows = num_samples // columns
    
    return (rows, columns)
    
def _image_grid(imgs, layout): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(layout[1]*w, layout[0]*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%layout[1]*w, i//layout[1]*h))
    return grid

# helper fft routines that keep ortho normalization and auto-shift before and after fft
def _fft2(data):
    if data.ndim > 2: # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft
   
def _ifft2(data):
    if data.ndim > 2: # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft
            
def _get_gaussian_window(width, height, std=3.14, mode=0):

    window_scale_x = float(width / min(width, height))
    window_scale_y = float(height / min(width, height))
    
    window = np.zeros((width, height))
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    for y in range(height):
        fy = (y / height * 2. - 1.) * window_scale_y
        if mode == 0:
            window[:, y] = np.exp(-(x**2+fy**2) * std)
        else:
            window[:, y] = (1/((x**2+1.) * (fy**2+1.))) ** (std/3.14) # hey wait a minute that's not gaussian
            
    return window

def _get_masked_window_rgb(np_mask_grey, hardness=1.):
    np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
    if hardness != 1.:
        hardened = np_mask_grey[:] ** hardness
    else:
        hardened = np_mask_grey[:]
    for c in range(3):
        np_mask_rgb[:,:,c] = hardened[:]
    return np_mask_rgb

"""
 Explanation:
 Getting good results in/out-painting with stable diffusion can be challenging.
 Although there are simpler effective solutions for in-painting, out-painting can be especially challenging because there is no color data
 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.
 Provided here is my take on a potential solution to this problem.
 
 By taking a fourier transform of the masked src img we get a function that tells us the presence and orientation of each feature scale in the unmasked src.
 Shaping the init noise for in/outpainting to the same distribution of feature scales, orientations, and positions increases output coherence
 by helping keep everything aligned. This technique is applicable to any continuous generation task such as audio or video, each of which can
 be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased. For multi-channel data such as color
 or stereo sound the "color tone" of the seed noise can also be matched to improve coherence (using scikit-image currently)
 This method is quite robust and has the added benefit of being fast independently of the size of the out-painted area.
 The effects of this method include things like helping the generator integrate the pre-existing view distance and camera angle.
 
 np_src_image is a float64 np array of shape [width,height,3] ( range 0..1)
 np_mask_rgb is a float64 np array of shape [width,height,3] ( range 0..1)
 noise_q controls the exponent in the fall-off of the distribution can be any positive number, lower values means higher detail (range > 0, default 1.)
 color_variation controls how much freedom is allowed for the colors/palette of the out-painted area (range 0..1, default 0.01)
 returns shaped noise for blending into the src image with the supplied mask ( [width,height,3] range 0..1 )
 
 This code is provided as is under the Unlicense (https://unlicense.org/)
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 This code is part of a new branch of a discord bot I am working on integrating with diffusers (https://github.com/parlance-zz/g-diffuser-bot)
 
"""
def _get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation): 

    global DEBUG_MODE
    global TMP_ROOT_PATH
    
    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2)/3.) 
    np_src_grey = (np.sum(np_src_image, axis=2)/3.) 
    all_mask = np.ones((width, height), dtype=bool)
    img_mask = np_mask_grey > 0.5
    ref_mask = np_mask_grey < 0.5
    
    windowed_image = _np_src_image * (1.-_get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb# / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
    #windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
    _save_debug_img(windowed_image, "windowed_src_img")
    
    src_fft = _fft2(windowed_image) # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist
    _save_debug_img(src_dist, "windowed_src_dist")
    
    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2)/3.) 
    noise_rgb *=  color_variation # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:,:,c] += (1. - color_variation) * noise_grey
        
    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:,:,c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:,:,:] = np.absolute(shaped_noise_fft[:,:,:])**2 * (src_dist ** noise_q) * src_phase # perform the actual shaping
    
    brightness_variation = 0.#color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.
    
    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    
    #shaped_noise -= np.min(shaped_noise)
    shaped_noise = np.clip(shaped_noise/ np.max(shaped_noise), 0., 1.)
    #shaped_noise_blended = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
    #shaped_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise_blended[all_mask,:], contrast_adjusted_np_src[ref_mask,:], channel_axis=1)
    shaped_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], contrast_adjusted_np_src[ref_mask,:], channel_axis=1)
    shaped_noise = np.clip(shaped_noise / np.max(shaped_noise), 0., 1.)
    _save_debug_img(shaped_noise, "shaped_noise")
    
    matched_noise = np.zeros((width, height, num_channels))
    matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    matched_noise[all_mask,:] = skimage.exposure.match_histograms(matched_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    #matched_noise[ref_mask,:] = skimage.exposure.match_histograms(matched_noise[ref_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    _save_debug_img(matched_noise, "matched_noise")
    
    """
    todo:
    color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be param controlled
    """
    
    return np.clip(matched_noise, 0., 1.) 
    
class CommandServer(BaseHTTPRequestHandler): # http command server

    def do_GET(self): # get command server status on GET
        
        try:
        
            self.send_response(200) # http OK
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            status_data = self.get_status()
            status_json = json.dumps(status_data, default=str)   
            self.wfile.write(bytes(status_json, "utf-8"))
            
        except Exception as e:
            print("Error sending status response - " + str(e) + "\n")

        return
        
    def do_POST(self): # execute command on POST
        
        try:
            post_body = self.rfile.read(int(self.headers['Content-Length']))
            post_body = post_body.decode("utf-8")
            cmd = json.loads(post_body)
        except Exception as e:
            print("Error in POST data - " + str(e))
            print(post_body + "\n")
            self.send_response(500) # http error
            return

        assert(type(cmd) == dict)
        
        try:

            self.send_response(200) # http OK
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            cmd = self.do_command(cmd)
            response_json = json.dumps(cmd, default=str)
            self.wfile.write(bytes(response_json, "utf-8"))
                    
        except Exception as e:
            raise
            print("Error sending command response - " + str(e) + "\n")
        
        return
        
    def do_command(self, _cmd):
     
        cmd = _cmd.copy()
        
        global TMP_ROOT_PATH
        
        global TXT2IMG_DIFFUSERS_PIPE
        global IMG2IMG_DIFFUSERS_PIPE
        global IMG_INP_DIFFUSERS_PIPE
        
        global MAX_RESOLUTION
        global MAX_STEPS_LIMIT
        global MAX_STRENGTH
        global MAX_OUTPUT_LIMIT
        
        start_time = datetime.datetime.now()
        cmd["start_time"] = str(start_time)
        
        # get params and set defaults
        try:
            num_samples = int(cmd["cmd_args"]["-n"]) if "-n" in cmd["cmd_args"] else 1
            init_image = cmd["in_attachments"][0] if len(cmd["in_attachments"]) > 0 else None
            strength = float(cmd["cmd_args"]["-str"]) if "-str" in cmd["cmd_args"] else 0.5
            guidance_scale = float(cmd["cmd_args"]["-scale"]) if "-scale" in cmd["cmd_args"] else None
            prompt = cmd["cmd_args"]["default_str"] if "default_str" in cmd["cmd_args"] else None
            width = int(cmd["cmd_args"]["-w"]) if "-w" in cmd["cmd_args"] else None
            height = int(cmd["cmd_args"]["-h"]) if "-h" in cmd["cmd_args"] else None
            num_inference_steps = int(cmd["cmd_args"]["-steps"]) if "-steps" in cmd["cmd_args"] else None
            color_variation = float(cmd["cmd_args"]["-color"]) if "-color" in cmd["cmd_args"] else 0.
            noise_q = float(cmd["cmd_args"]["-noise_q"]) if "-noise_q" in cmd["cmd_args"] else 0.99
            mask_blend_factor = float(cmd["cmd_args"]["-blend"]) if "-blend" in cmd["cmd_args"] else 1.
        except Exception as e:
            cmd["status"] = -1 # error status
            cmd["error_txt"] = "Error getting params '" + str(e) + "'"
            print(cmd["error_txt"] + "\n")
            return cmd
        
        pipe = TXT2IMG_DIFFUSERS_PIPE
        
        # validate params
        try:
            if num_inference_steps:
                num_inference_steps = min(num_inference_steps, MAX_STEPS_LIMIT)
            if strength:
                strength = max(min(strength, MAX_STRENGTH), 0.0)
            if guidance_scale:
                guidance_scale = max(guidance_scale, 0.0)
            if num_samples:
                num_samples = min(num_samples, MAX_OUTPUT_LIMIT)
            if mask_blend_factor < 1.:
                mask_blend_factor = 1.
                
            if init_image:
                try:
                    init_image = Image.open(init_image)
                except Exception as e:
                    raise 
                    cmd["status"] = -1 # error status
                    cmd["error_txt"] = "Error loading img or mask '" + str(e) + "'"
                    print(cmd["error_txt"] + "\n")
                    return cmd

            width, height = _valid_resolution(width, height, init_image=init_image)
            
            if init_image:
                try:
                    if (width, height) != init_image.size: # default size is native img size
                        print("Resizing input img to (" + str(width) + ", " + str(height) + ")")    
                        init_image = init_image.resize((width, height), resample=PIL.Image.LANCZOS)

                    # extract mask_image from alpha
                    if init_image.mode == "RGBA":
                        mask_image = init_image.split()[-1]
                        mask_image = PIL.ImageOps.invert(mask_image)
                        init_image = init_image.convert("RGB")
                        assert(mask_image.size == init_image.size)
                        
                        if not mask_image.getbbox(): # if mask is all opaque anyway just use regular img2img pipe
                            mask_image = None
                        else:
                            np_init = (np.asarray(init_image.convert("RGB"))/255.0).astype(np.float64)
                            np_mask_rgb = (np.asarray(mask_image.convert("RGB"))/255.0).astype(np.float64)
                            
                    else:
                        mask_image = None
                        
                    if mask_image: # choose a pipe
                        pipe = IMG_INP_DIFFUSERS_PIPE
                    else:
                        pipe = IMG2IMG_DIFFUSERS_PIPE

                except Exception as e:
                    raise 
                    cmd["status"] = -1 # error status
                    cmd["error_txt"] = "Error loading img or mask '" + str(e) + "'"
                    print(cmd["error_txt"] + "\n")
                    return cmd
            
            else: # txt2img
                print("")
                        
        except Exception as e:
            raise
            cmd["status"] = -1 # error status
            cmd["error_txt"] = "Error checking params '" + str(e) + "'"
            print(cmd["error_txt"] + "\n")
            return cmd
            
        with autocast("cuda"):
            try:
                # finally generate the actual samples
                samples = []
                for i in range(num_samples):
                    if pipe == IMG2IMG_DIFFUSERS_PIPE:
                        print("Using img2img pipeline...")
                        sample = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
                    elif pipe == IMG_INP_DIFFUSERS_PIPE:
                        print("Using img in-painting pipeline...")
                        
                        noise_rgb = _get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
                        blend_mask_rgb = ((np_mask_rgb*2.) ** mask_blend_factor) / (2. ** mask_blend_factor)
                        noised = np_init[:] * (1. - blend_mask_rgb) + noise_rgb * blend_mask_rgb
                        
                        # one last thing, gotta colorize the noise from src while preserving vector mag of blended noise img
                        """
                        noised_mag = np.sum(noise_rgb**2, axis=2) ** 0.5
                        noised_mag_rgb = np.zeros((noised_mag.shape[0], noised_mag.shape[1], 3))
                        for c in range(3):
                            noised_mag_rgb[:,:,c] = noised_mag[:]
                        colorize_mask = (blend_mask_rgb > 0.99).astype(np.float64)                                              # todo: magic constant
                        noised_colorized = noised_mag_rgb * np_init[:,:,:] ** (((1. - blend_mask_rgb) ** 0.1) * colorize_mask) # todo: derive magic constant
                        """
                        #"""
                        for t in range(8): # 4
                            noised = np_init[:] * (1. - blend_mask_rgb) + noised * blend_mask_rgb # and blend back to the src again
                            #noised = np.clip(noised, 0. , 1.)
                        #"""
                        
                        init_image = PIL.Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")

                        _save_debug_img(init_image, "init_img_" + str(i+1))
                        _save_debug_img(mask_image, "mask_img_" + str(i+1))                        
                        _save_debug_img(blend_mask_rgb, "blend_mask_rgb_" + str(i+1))
                        
                        sample = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, mask_image=mask_image, num_inference_steps=num_inference_steps)
                    else:
                        print("Using txt2img pipeline...")
                        sample = pipe(prompt=prompt, guidance_scale=guidance_scale, width=width, height=height, num_inference_steps=num_inference_steps)
                    
                    samples.append(sample["sample"][0])
                    
            except Exception as e:
                raise
                cmd["status"] = -1 # error status
                cmd["error_txt"] = "Error in diffuser pipeline '" + str(e) + "'"
                print(cmd["error_txt"] + "\n")
        
        if cmd["status"] != -1: # if no error so far, save outputs and set success status
            try:
                cmd["out_resolution"] = (width, height)
                sample_format = ".png"
                if (width * height) > (1024 * 1024):
                    sample_format = ".jpg"
                    
                for sample in samples: # save each output individually
                    output_path = _get_tmp_path(sample_format)
                    sample.save(output_path)
                    cmd["out_attachments"].append(output_path)
                
                if len(samples) > 1: # if more than one output, make the preview image a single grid image of all of them
                    cmd["out_preview_image_layout"] = _get_grid_layout(len(samples))
                    grid_image = _image_grid(samples, cmd["out_preview_image_layout"])
                    
                    out_preview_sample_format = ".png"
                    if (cmd["out_preview_image_layout"][0] * width * cmd["out_preview_image_layout"][1] * height) > (1024 * 1024):
                        out_preview_sample_format = ".jpg"
                        
                    cmd["out_preview_image"] = _get_tmp_path(out_preview_sample_format)
                    grid_image.save(cmd["out_preview_image"])
                else:
                    cmd["out_preview_image"] = cmd["out_attachments"][0]
                    cmd["out_preview_image_layout"] = (1, 1)
                
                cmd["status"] = 2 # run successfully
                cmd["elapsed_time"] = str(datetime.datetime.now() - start_time)
            except Exception as e:
                cmd["status"] = -1 # error status
                cmd["error_txt"] = "Error saving output '" + str(e) + "'"
                print(cmd["error_txt"])
                
        return cmd
        
    def get_status(self):
        status = { "ok": True }
        return status
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_server', dest='start_server', action='store_true')
    args = parser.parse_args()
    
    if args.start_server:
    
        print("Loading diffuser pipelines...")
        
        revision = None
        torch_dtype = None
        scheduler = None
        
        if BOT_USE_OPTIMIZED: # optimized version uses fp16 to reduce memory consumption
            revision = "fp16"
            torch_dtype = torch.float16
        
        """
        if CMD_SERVER_MODEL_SCHEDULER.lower() == "lms": # optional LMS sampler support
            print("Using LMS noise scheduler with diffuser pipelines...")
            scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear"
            )
        """
        
        # create pipes
        print("Loading txt2img pipeline...")
        TXT2IMG_DIFFUSERS_PIPE = StableDiffusionPipeline.from_pretrained(
            CMD_SERVER_MODEL_NAME, 
            revision=revision, 
            torch_dtype=torch_dtype,
        #    scheduler=scheduler,
            use_auth_token=HUGGINGFACE_TOKEN
        )
        TXT2IMG_DIFFUSERS_PIPE = TXT2IMG_DIFFUSERS_PIPE.to("cuda")
        
        print("Loading img2img pipeline...")
        IMG2IMG_DIFFUSERS_PIPE = StableDiffusionImg2ImgPipeline.from_pretrained(
            CMD_SERVER_MODEL_NAME, 
            revision=revision, 
            torch_dtype=torch_dtype,
        #    scheduler=scheduler,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        IMG2IMG_DIFFUSERS_PIPE = IMG2IMG_DIFFUSERS_PIPE.to("cuda")
        
        print("Loading img_inp pipeline...")
        IMG_INP_DIFFUSERS_PIPE = StableDiffusionInpaintPipeline.from_pretrained(
            CMD_SERVER_MODEL_NAME, 
            revision=revision, 
            torch_dtype=torch_dtype,
        #    scheduler=scheduler,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        IMG_INP_DIFFUSERS_PIPE = IMG_INP_DIFFUSERS_PIPE.to("cuda")
    
        # if optimized use attention slicing for lower memory consumption
        if BOT_USE_OPTIMIZED:
            TXT2IMG_DIFFUSERS_PIPE.enable_attention_slicing()
            IMG2IMG_DIFFUSERS_PIPE.enable_attention_slicing()
            IMG_INP_DIFFUSERS_PIPE.enable_attention_slicing()
        
        # replace safety checkers with dummy
        setattr(TXT2IMG_DIFFUSERS_PIPE, "safety_checker", _dummy_checker)
        setattr(IMG2IMG_DIFFUSERS_PIPE, "safety_checker", _dummy_checker)
        setattr(IMG_INP_DIFFUSERS_PIPE, "safety_checker", _dummy_checker)
    
        #"""
        # reduce memory consumption (https://gist.github.com/fladdict/2115eb7ea32c9245e4f45642553aa3e9)
        IMG2IMG_DIFFUSERS_PIPE.vae = IMG_INP_DIFFUSERS_PIPE.vae = TXT2IMG_DIFFUSERS_PIPE.vae
        IMG2IMG_DIFFUSERS_PIPE.text_encoder = IMG_INP_DIFFUSERS_PIPE.text_encoder = TXT2IMG_DIFFUSERS_PIPE.text_encoder
        IMG2IMG_DIFFUSERS_PIPE.tokenizer = IMG_INP_DIFFUSERS_PIPE.tokenizer = TXT2IMG_DIFFUSERS_PIPE.tokenizer
        IMG2IMG_DIFFUSERS_PIPE.unet = IMG_INP_DIFFUSERS_PIPE.unet = TXT2IMG_DIFFUSERS_PIPE.unet
        IMG2IMG_DIFFUSERS_PIPE.feature_extractor = IMG_INP_DIFFUSERS_PIPE.feature_extractor = TXT2IMG_DIFFUSERS_PIPE.feature_extractor
        IMG2IMG_DIFFUSERS_PIPE.scheduler = IMG_INP_DIFFUSERS_PIPE.scheduler = TXT2IMG_DIFFUSERS_PIPE.scheduler
        #"""
        
        print("Finished loading diffuser pipelines... Starting local command server...")
        web_server = HTTPServer((CMD_SERVER_BIND_HTTP_HOST, CMD_SERVER_BIND_HTTP_PORT), CommandServer)
        print("CommandServer started http://" + CMD_SERVER_BIND_HTTP_HOST + ":" + str(CMD_SERVER_BIND_HTTP_PORT))
        try:
            web_server.serve_forever()
        except KeyboardInterrupt:
            pass
        
        web_server.server_close()
        
    else:
    
        print("Please run python g_diffuser_bot.py to start the G-Diffuser-Bot")