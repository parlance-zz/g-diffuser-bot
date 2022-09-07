# todo:
# - setting seeds doesnt work
# - add !enhance support back in
# - outpainting
# - add style transfer function that takes input image and 50% erases it
# - check for command cancelling between individual pipe calls / samples

#  ??. integrate gfp-gan to fix faces

from g_diffuser_bot_params import *

import os, sys
os.chdir(ROOT_PATH)

import argparse
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
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

def _get_tmp_path(file_extension):
    global TMP_ROOT_PATH
    try: # try to make sure temp folder exists
        pathlib.Path(TMP_ROOT_PATH).mkdir(exist_ok=True)
    except Exception as e:
        print("Error creating temp path: '" + TMP_ROOT_PATH + "' - " + str(e))
    return TMP_ROOT_PATH + "/" + str(uuid.uuid4()) + file_extension
    
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

def _premultiply_alpha(im):
    pixels = im.load()
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            r, g, b, a = pixels[x, y]
            if a != 255:
                r = r * a // 255
                g = g * a // 255
                b = b * a // 255
                pixels[x, y] = (r, g, b, a)
                
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
        
        # get params
        try:
            num_samples = int(cmd["cmd_args"]["-n"]) if "-n" in cmd["cmd_args"] else 1
            init_image = cmd["in_attachments"][0] if len(cmd["in_attachments"]) > 0 else None
            strength = float(cmd["cmd_args"]["-str"]) if "-str" in cmd["cmd_args"] else 0.4
            guidance_scale = float(cmd["cmd_args"]["-scale"]) if "-scale" in cmd["cmd_args"] else None
            prompt = cmd["cmd_args"]["default_str"] if "default_str" in cmd["cmd_args"] else None
            width = int(cmd["cmd_args"]["-w"]) if "-w" in cmd["cmd_args"] else None
            height = int(cmd["cmd_args"]["-h"]) if "-h" in cmd["cmd_args"] else None
            num_inference_steps = int(cmd["cmd_args"]["-steps"]) if "-steps" in cmd["cmd_args"] else None
        except Exception as e:
            cmd["status"] = -1 # error status
            cmd["error_txt"] = "Error getting params '" + str(e) + "'"
            print(cmd["error_txt"] + "\n")
            return cmd
        
        pipe = TXT2IMG_DIFFUSERS_PIPE
        
        # check params
        try:
            if width:
                width = min(width, MAX_RESOLUTION[0])
            if height:
                height = min(height, MAX_RESOLUTION[1])
            if num_inference_steps:
                num_inference_steps = min(num_inference_steps, MAX_STEPS_LIMIT)
            if strength:
                strength = max(min(strength, MAX_STRENGTH), 0.0)
            if guidance_scale:
                guidance_scale = max(guidance_scale, 0.0)
            if num_samples:
                num_samples = min(num_samples, MAX_OUTPUT_LIMIT)
                
            mask_image = None
            if init_image:
                try:
                    init_image = Image.open(init_image)
                    
                    if not width:
                        width = init_image.size[0]
                    if not height:
                        height = init_image.size[1]
                    
                    # input images must be sized to a multiple of 32, additionally we set a minimum size

                    n_width = int(int(width / 32. + 0.5) * 32)
                    n_height = int(int(height / 32. + 0.5) * 32)
                    if n_width < 32: n_width = 32
                    if n_height < 32: n_height = 32
                    if (n_width != width) or (n_height != height) or (init_image.size != (width,height)):
                        print("Resizing img to (" + str(width) + ", " + str(height) + ")")
                        
                        # due to a stupid bug in PIL we have to manually pre-multiply the colors by alpha before resize if we have an alpha channel
                        #if init_image.mode == "RGBA":
                        #    _premultiply_alpha(init_image)
                            
                        init_image = init_image.resize((n_width, n_height), resample=PIL.Image.LANCZOS)
                        width = n_width
                        height = n_height
                    
                    # make mask_image
                    if init_image.mode == "RGBA":
                        mask_image = init_image.split()[-1]
                        mask_image = PIL.ImageOps.invert(mask_image)
                        init_image = init_image.convert("RGB")
                        assert(mask_image.size == init_image.size)
                        
                        if not mask_image.getbbox(): # if mask is all opaque anyway just use regular img2img pipe
                            mask_image = None
                        else:
                            # add noise to erased area inversely proportional to str
                            # todo: this is a total hack job
                            np_init = (np.asarray(init_image.convert("RGB"))/255.0).astype(np.float64)
                            np_mask_a = (np.asarray(mask_image.convert("L"))/255.0).astype(np.float64)
                            np_mask_rgb = (np.asarray(mask_image.convert("RGB"))/255.0).astype(np.float64)
                            mean_r = np.sum(np_init[:,:,0]) / width / height
                            mean_g = np.sum(np_init[:,:,1]) / width / height
                            mean_b = np.sum(np_init[:,:,2]) / width / height
                            var_r = np.sum((np_init[:,:,0] - mean_r)**2) / width / height 
                            var_g = np.sum((np_init[:,:,1] - mean_g)**2) / width / height
                            var_b = np.sum((np_init[:,:,2] - mean_b)**2) / width / height
                            
                            assert(strength != None)
                            
                            # todo: too many magic numbers
                            color_tone_freedom = (strength ** 2) * 0.12
                            num_blend_steps = 3
                            blend_step_radius = 83.8 # 61.8
                            blend_step_final_radius = 0.618
                            
                            noise_factor = 0.2

                            if strength > 0.98:
                                strength = 0.81
                                noise_factor = 0.69               
                            elif strength > 0.91:
                                strength = 0.74
                                noise_factor = 0.57
                            elif strength > 0.79:
                                strength = 0.66
                                noise_factor = 0.44
                            elif strength > 0.58:
                                strength = 0.52
                                noise_factor = 0.31

                            var_r *= 1. + color_tone_freedom
                            var_g *= 1. + color_tone_freedom
                            var_b *= 1. + color_tone_freedom
                            np_mask_a_scaled = np_mask_a[:] * noise_factor
                            np_mask_rgb_scaled = np_mask_rgb[:] * noise_factor
                            
                            noised = np_init[:].copy()
                            noised[:,:,0] = mean_r
                            noised[:,:,1] = mean_g
                            noised[:,:,2] = mean_b
                            
                            for i in range(num_blend_steps):
                                noise_r = np.random.normal(0.,var_r**0.5,(width,height))
                                noise_g = np.random.normal(0.,var_g**0.5,(width,height))
                                noise_b = np.random.normal(0.,var_b**0.5,(width,height))
                                noised[:,:,0] += noise_r
                                noised[:,:,1] += noise_g
                                noised[:,:,2] += noise_b
                                noised_image = PIL.Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")
                                if (i < (num_blend_steps-1)):
                                    noised_image = noised_image.filter(ImageFilter.GaussianBlur(radius=blend_step_radius))
                                    noised = np.asarray(noised_image.convert("RGB"))/255.
                                else:
                                    if blend_step_final_radius > 0:
                                        noised_image = noised_image.filter(ImageFilter.GaussianBlur(radius=blend_step_final_radius))
                                        noised = np.asarray(noised_image.convert("RGB"))/255.
                                        
                                noised[:] = np_init[:] * (1. - np_mask_rgb_scaled) + noised[:] * np_mask_rgb_scaled
                            
                            init_image = PIL.Image.fromarray((noised.clip(0., 1.)*255.).astype(np.uint8), mode="RGB")
                            try: # these are helpful for debugging in-painting
                                init_image.save(TMP_ROOT_PATH + "/_debug_init_img.png")
                                mask_image.save(TMP_ROOT_PATH + "/_debug_mask_img.png")
                            except Exception as e:
                                print("Error saving debug images - " + str(e))
                    
                    if mask_image: # choose a pipe
                        pipe = IMG_INP_DIFFUSERS_PIPE
                        
                    else:
                        pipe = IMG2IMG_DIFFUSERS_PIPE

                except Exception as e:
                    cmd["status"] = -1 # error status
                    cmd["error_txt"] = "Error loading img or mask '" + str(e) + "'"
                    print(cmd["error_txt"] + "\n")
                    return cmd
                
        except Exception as e:
            cmd["status"] = -1 # error status
            cmd["error_txt"] = "Error checking params '" + str(e) + "'"
            print(cmd["error_txt"] + "\n")
            return cmd
            
        with autocast("cuda"):
            try:
                """
                try:
                    pipe.to("cuda") # use gpu pls
                except:
                    print("Error moving model to cuda in pipe.to() \n")
                """

                # finally generate the actual samples
                samples = []
                for i in range(num_samples):
                    if pipe == IMG2IMG_DIFFUSERS_PIPE:
                        print("Using img2img pipeline...")
                        sample = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
                    elif pipe == IMG_INP_DIFFUSERS_PIPE:
                        print("Using img in-painting pipeline...")
                        sample = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, mask_image=mask_image, num_inference_steps=num_inference_steps)
                    else:
                        print("Using txt2img pipeline...")
                        sample = pipe(prompt=prompt, guidance_scale=guidance_scale, width=width, height=height, num_inference_steps=num_inference_steps)
                    
                    samples.append(sample["sample"][0])
                    
            except Exception as e:
                cmd["status"] = -1 # error status
                cmd["error_txt"] = "Error in diffuser pipeline '" + str(e) + "'"
                print(cmd["error_txt"] + "\n")
            
            """
            try:
                pipe.to("cpu") # reclaim gpu memory
            except:
                print("Error moving model to cpu in pipe.to() \n")
            """
        
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


"""
    elif cmd.command in ["!enhance"]:
    
        try:
            cmd.status = 1 # in-progress
            cmd.start_time = datetime.datetime.now()
            
            input_path = cmd.in_image
            if input_path == "": input_path = CMD_QUEUE.get_last_attached(user=cmd.init_user)
            if input_path == "":
                cmd.error_txt = "No attached image"
                cmd.status = -1 # error
            else:
            
                if not input_path.endswith(".png"):
                    img = Image.open(input_path)
                    tmp_file_path = _get_tmp_path(".png")
                    img.save(tmp_file_path)
                    img.close()
                    input_path = tmp_file_path

                enhanced_file_path = _get_tmp_path(".png")
                cmd.cmd_args = _merge_dicts(ESRGAN_ADD_PARAMS, cmd.cmd_args)
                cmd.cmd_args["-i"] = '"' + input_path + '"'
                cmd.cmd_args["-o"] = '"' + enhanced_file_path + '"'
                
                cmd.run_string = _join_args(ESRGAN_PATH, cmd.cmd_args)
                return_code = await _run_cmd(cmd.run_string, cmd)
                
                if cmd.status != 3: # cancelled
                    if return_code != 0:
                        cmd.error_txt = "Error returned by command"
                        cmd.status = -1  # error
                    else:
                        print(enhanced_file_path)
                        cmd.out_image = enhanced_file_path
                        cmd.out_resolution = (512 * 4, 512 * 4)
                        cmd.out_preview_image_layout = (1, 1)
                        cmd.status = 2 # success
"""
