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
"""

from g_diffuser_bot_defaults import *
from g_diffuser_lib import *

import os, sys
os.chdir(ROOT_PATH)

import argparse
import datetime
import asyncio
import json
import pathlib
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer


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
            self.send_response(500) # http generic server error
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
        global DEFAULT_STRENGTH
        global MAX_OUTPUT_LIMIT
        
        start_time = datetime.datetime.now()
        cmd["start_time"] = str(start_time)
        
        # get params and set defaults
        try:
            num_samples = int(cmd["cmd_args"]["-n"]) if "-n" in cmd["cmd_args"] else 1
            init_image = cmd["in_attachments"][0] if len(cmd["in_attachments"]) > 0 else None
            strength = float(cmd["cmd_args"]["-str"]) if "-str" in cmd["cmd_args"] else 0.
            guidance_scale = float(cmd["cmd_args"]["-scale"]) if "-scale" in cmd["cmd_args"] else None
            prompt = cmd["cmd_args"]["default_str"] if "default_str" in cmd["cmd_args"] else None
            width = int(cmd["cmd_args"]["-w"]) if "-w" in cmd["cmd_args"] else None
            height = int(cmd["cmd_args"]["-h"]) if "-h" in cmd["cmd_args"] else None
            num_inference_steps = int(cmd["cmd_args"]["-steps"]) if "-steps" in cmd["cmd_args"] else None
            color_variation = float(cmd["cmd_args"]["-color"]) if "-color" in cmd["cmd_args"] else 0.
            noise_q = float(cmd["cmd_args"]["-noise_q"]) if "-noise_q" in cmd["cmd_args"] else 1.
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
                num_inference_steps = np.minimum(num_inference_steps, MAX_STEPS_LIMIT)
            if strength:
                strength = np.clip(strength, 0., MAX_STRENGTH)
            if guidance_scale:
                guidance_scale = np.maximum(guidance_scale, 0.0)
            if num_samples:
                num_samples = min(num_samples, MAX_OUTPUT_LIMIT)
            if mask_blend_factor < 1e-4:
                mask_blend_factor = 1e-4
                
            if init_image:
                try:
                    init_image = Image.open(init_image)
                except Exception as e:
                    raise 
                    cmd["status"] = -1 # error status
                    cmd["error_txt"] = "Error loading img or mask '" + str(e) + "'"
                    print(cmd["error_txt"] + "\n")
                    return cmd

            # find a valid resolution (multiple of 64) under max res, while trying to maintain aspect ratio
            width, height = _valid_resolution(width, height, init_image=init_image)
            
            if init_image:
                try:
                    if (width, height) != init_image.size: # default size is native img size
                        print("Resizing input image to (" + str(width) + ", " + str(height) + ")")    
                        init_image = init_image.resize((width, height), resample=PIL.Image.LANCZOS)

                    # extract mask_image from alpha
                    if init_image.mode == "RGBA":
                    
                        mask_image = init_image.split()[-1]
                        init_image = init_image.convert("RGB")
                        
                        if not mask_image.getbbox(): # if mask is all opaque anyway just use regular img2img pipe
                            mask_image = None
                        else:
                            np_init = (np.asarray(init_image.convert("RGB"))/255.).astype(np.float64)
                            np_mask_rgb = (np.asarray(mask_image.convert("RGB"))/255.).astype(np.float64)
                            if np.min(np_mask_rgb) > 0.:
                                print("Warning: Image mask doesn't have any fully transparent area")
                            if np.max(np_mask_rgb) < 1.:
                                print("Warning: Image mask doesn't have any opaque area")
                            
                            if strength > 0.:
                                print("Warning: Overriding mask maximum opacity with supplied strength : " + str(strength))
                                
                            mask_hardened, final_blend_mask, window_mask = _get_blend_masks(np_mask_rgb, mask_blend_factor, strength) # annoyingly complex mask manipulation
                            mask_image = PIL.Image.fromarray(np.clip(final_blend_mask*255., 0., 255.).astype(np.uint8), mode="RGB")
                            
                    else:
                        mask_image = None
                        if strength == 0.: strength = DEFAULT_STRENGTH
                        
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
                        sample = pipe(
                            prompt=prompt,
                            init_image=init_image,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps
                        )
                    elif pipe == IMG_INP_DIFFUSERS_PIPE:
                        print("Using img in-painting pipeline...")
                        noised = _get_matched_noise(np_init, mask_hardened, final_blend_mask, window_mask, noise_q)
                        init_image = PIL.Image.fromarray(np.clip(noised*255., 0., 255.).astype(np.uint8), mode="RGB")
                        
                        sample = pipe(
                            prompt=prompt,
                            init_image=init_image,
                            strength=MAX_STRENGTH,  # this is important when using masked img2img
                            guidance_scale=guidance_scale,
                            mask_image=mask_image,
                            num_inference_steps=num_inference_steps
                        )
                    else:
                        print("Using txt2img pipeline...")
                        sample = pipe(
                            prompt=prompt,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps
                        )
                    
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
                if (width * height) > (1024 * 1024): # who wants to pay for nitro
                    sample_format = ".jpg"
                    
                for sample in samples: # save each output individually
                    output_path = _get_tmp_path(sample_format)
                    sample.save(output_path)
                    cmd["out_attachments"].append(output_path)
                
                if len(samples) > 1: # if more than one output, make the preview image a single grid image of all of them
                    cmd["out_preview_image_layout"] = _get_grid_layout(len(samples))
                    grid_image = _get_image_grid(samples, cmd["out_preview_image_layout"])
                    
                    out_preview_sample_format = ".png"
                    if (cmd["out_preview_image_layout"][0] * width * cmd["out_preview_image_layout"][1]*height) > (1024*1024):
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
        loaded_pipes = _load_pipelines()
        
        print("Starting command server...")
        web_server = HTTPServer((CMD_SERVER_BIND_HTTP_HOST, CMD_SERVER_BIND_HTTP_PORT), CommandServer)
        print("CommandServer started successfully at http://" + CMD_SERVER_BIND_HTTP_HOST + ":" + str(CMD_SERVER_BIND_HTTP_PORT))
        try:
            web_server.serve_forever()
        except KeyboardInterrupt:
            pass
        web_server.server_close()
        
    else:
        print("Please run python g_diffuser_bot.py to start the G-Diffuser-Bot")