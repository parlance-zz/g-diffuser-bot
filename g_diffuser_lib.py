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

from g_diffuser_config import DEFAULT_PATHS, GRPC_SERVER_SETTINGS
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS

import os, sys
import io
import time
import datetime
import argparse
import uuid
import pathlib
import json
import re
import importlib
import subprocess
import psutil

import numpy as np
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

from extensions import grpc_client

import torch
from torch import autocast

def _p_kill(proc_pid):  # kill all child processes, recursively as well. its the only way to be sure
    print("Killing process id " + str(proc_pid))
    try:
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True): proc.kill()
        process.kill()
    except Exception as e: print("Error killing process id " + str(proc_pid) + " - " + str(e))
    return
    
def run_string(run_string, cwd, show_output=False, log_path=""):  # run shell command asynchronously, return subprocess
    print(run_string + " (cwd="+str(cwd)+")")
    if log_path != "": process = subprocess.Popen(run_string, shell=False, cwd=cwd, stdout=open(log_path, "w", 1))
    else: process = subprocess.Popen(run_string, shell=False, cwd=cwd)
    assert(process)
    return process
    
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

def print_namespace(namespace, debug=False, verbosity_level=0, indent=4):
    namespace_dict = vars(strip_args(namespace, level=verbosity_level))
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
        if np_image.ndim == 2: mode = "L"
        elif np_image.shape[2] == 4: mode = "RGBA"
        else: mode = "RGB"
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
    
def load_json(file_path):
    assert(file_path)
    (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True)
    
    with open(file_path, "r") as file:
        data = json.load(file)
        file.close()
    return data
    
def strip_args(args, level=0): # remove args we wouldn't want to print or serialize, higher levels strip additional irrelevant fields
    args_stripped = argparse.Namespace(**(vars(args).copy()))
    if "grpc_server_process" in args_stripped: del args_stripped.grpc_server_process
    
    if level >=1: # keep just the basics for most printing
        if "command" in args_stripped: del args_stripped.command
        if "seed" in args_stripped: del args_stripped.seed
        if "use_optimized" in args_stripped: del args_stripped.use_optimized
        if "debug" in args_stripped: del args_stripped.debug
        if "interactive" in args_stripped: del args_stripped.interactive
        if "load_args" in args_stripped: del args_stripped.load_args
        if "no_json" in args_stripped: del args_stripped.no_json
        if "hf_token" in args_stripped: del args_stripped.hf_token
        if "init_time" in args_stripped: del args_stripped.init_time
        if "start_time" in args_stripped: del args_stripped.start_time
        if "end_time" in args_stripped: del args_stripped.end_time
        if "elapsed_time" in args_stripped: del args_stripped.elapsed_time
        if "output" in args_stripped: del args_stripped.output
        if "args_output" in args_stripped: del args_stripped.args_output
        if "output_samples" in args_stripped: del args_stripped.output_samples
        if "uuid_str" in args_stripped: del args_stripped.uuid_str
        if "status" in args_stripped: del args_stripped.status
        if "used_pipe" in args_stripped: del args_stripped.used_pipe
        if "outputs_path" in args_stripped: del args_stripped.outputs_path
        if "n" in args_stripped: del args_stripped.n
        
        if "init_img" in args_stripped:
            if args_stripped.init_img == "": # if there was no input image these fields are also irrelevant
                if "init_img" in args_stripped: del args_stripped.init_img
                if "noise_q" in args_stripped: del args_stripped.noise_q
                if "strength" in args_stripped: del args_stripped.strength
                
    return args_stripped
    
def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]
    
def get_grid_layout(num_samples):
    factors = factorize(num_samples)
    median_factor = factors[len(factors)//2]
    columns = median_factor
    rows = num_samples // columns
    return (rows, columns)
    
def get_image_grid(imgs, layout, mode=0): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(layout[1]*w, layout[0]*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        if mode: grid.paste(img, box=(i//layout[0]*w, ilayout[0]*h))
        else: grid.paste(img, box=(i%layout[1]*w, i//layout[1]*h))
    return grid

def load_image(args):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS
    assert(DEFAULT_PATHS.inputs)
    final_init_img_path = (pathlib.Path(DEFAULT_PATHS.inputs) / args.init_img).as_posix()
    
    # load and resize input image to multiple of 8x8
    init_image = Image.open(final_init_img_path)
    width, height = valid_resolution(args.w, args.h, init_image=init_image)
    if (width, height) != init_image.size:
        if args.debug: print("Resizing input image to (" + str(width) + ", " + str(height) + ")")
        init_image = init_image.resize((width, height), resample=PIL.Image.LANCZOS)
    args.w = width
    args.h = height
        
    if init_image.mode == "RGBA": # in/out-painting
        # prep masks, note that you only need to prep masks once if you're doing multiple samples
        mask_image = init_image.split()[-1]
        np_mask_rgb = (np.asarray(mask_image.convert("RGB"))/255.).astype(np.float64)
        mask_image = PIL.Image.fromarray(np.clip(np_mask_rgb*255., 0., 255.).astype(np.uint8), mode="RGB")

    else: # img2img
        if args.strength == 0.: args.strength = DEFAULT_SAMPLE_SETTINGS.strength
        blend_mask = np_img_grey_to_rgb(np.ones((args.w, args.h)) * np.clip(args.strength**(0.075), 0., 1.)) # todo: find strength mapping or do a better job of seeding
        mask_image = PIL.Image.fromarray(np.clip(blend_mask*255., 0., 255.).astype(np.uint8), mode="RGB")

    return init_image, mask_image
        
def get_samples(args):
    global DEFAULT_PATHS
    global DEFAULT_SAMPLE_SETTINGS, GRPC_SERVER_SETTINGS
    
    #if args.debug:
    #    importlib.reload(ext) # this allows us to test modifications to extensions without reloading the cli or model
    
    args.uuid_str = get_random_string(digits=16) # new uuid for new sample(s)
    args.status = 1 # running
    
    if args.init_img != "": # load input image if we have one
        init_image, mask_image = load_image(args)
    else:
        init_image, mask_image = (None, None)
        if not args.w: args.w = DEFAULT_SAMPLE_SETTINGS.resolution[0] # if we don't have an input image, it's size can't be used as the default resolution
        if not args.h: args.h = DEFAULT_SAMPLE_SETTINGS.resolution[1]
        
    
    start_time = datetime.datetime.now()
    args.start_time = str(start_time)
    error_in_sampling = False
    samples = []
    
    for n in range(args.n): 
        if args.status == 3: return # if command is cancelled just bail out asap
        try:
            
            request_dict = build_grpc_request_dict(args)
            stability_api = grpc_client.StabilityInference(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.key, engine=args.model_name, verbose=False)
    
            answers = stability_api.generate(args.prompt, **request_dict)
            output_prefix = DEFAULT_PATHS.temp+"/"
            grpc_samples = grpc_client.process_artifacts_from_answers(output_prefix, answers, write=True, verbose=False)
            for path, artifact in grpc_samples:
                image = Image.open(io.BytesIO(artifact.binary))
                samples.append(image)
            
            """
            sample = pipe(
                prompt=args.prompt,
                guidance_scale=args.scale,
                num_inference_steps=args.steps,
                width=args.w,
                height=args.h,
                init_image=init_image,
                mask_image=mask_image,
                strength=args.strength,
            )
            """
        except Exception as e:
            raise
            error_in_sampling = True
            args.status = -1
            return []
            
    end_time = datetime.datetime.now()
    args.end_time = str(end_time)
    args.elapsed_time = str(end_time-start_time)
    
    args.status = 2 # completed successfully
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
    
def start_grpc_server(args):
    global DEFAULT_PATHS, GRPC_SERVER_SETTINGS
    if args.debug: load_start_time = datetime.datetime.now()
    
    if DEFAULT_PATHS.grpc_log != DEFAULT_PATHS.root:
        log_path = DEFAULT_PATHS.grpc_log
    else:
        log_path = ""
        
    grpc_server_run_string = "python ./server.py"
    grpc_server_run_string += " --enginecfg "+DEFAULT_PATHS.root+"/g_diffuser_config_models.yaml" + " --weight_root "+DEFAULT_PATHS.models
    grpc_server_run_string += " --vram_optimisation_level " + str(GRPC_SERVER_SETTINGS.memory_optimization_level)
    if GRPC_SERVER_SETTINGS.enable_mps: grpc_server_run_string += " --enable_mps"
    grpc_server_process = run_string(grpc_server_run_string, cwd=DEFAULT_PATHS.extensions+"/"+"stable-diffusion-grpcserver", log_path=log_path)
    
    if args.debug: print("sd_grpc_server start time : " + str(datetime.datetime.now() - load_start_time))
    
    args.grpc_server_process = grpc_server_process
    return grpc_server_process
   
# todo: merge with my init args to allow user override without editing files
"""
def get_gprc_args_parser():
    # add gprc server args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--enginecfg", "-E", type=str, default="./extensions/gprc_engines.yaml", help="Path to the engines.yaml file"
    )
    parser.add_argument(
        "--listen_to_all", "-L", action='store_true', help="Accept requests from the local network, not just localhost" 
    )
    parser.add_argument(
        "--enable_mps", type=bool, default=False, help="Use MPS on MacOS where available"
    )
    parser.add_argument(
        "--vram_optimisation_level", "-V", type=int, default=2, help="How much to trade off performance to reduce VRAM usage (0 = none, 2 = max)"
    )
    parser.add_argument(
        "--nsfw_behaviour", "-N", type=str, default="block", choices=["block", "flag"], help="What to do with images detected as NSFW"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on source change"
    )
"""
 
def get_args_parser():
    global DEFAULT_SAMPLE_SETTINGS
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the text to condition sampling on",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_SAMPLE_SETTINGS.model_name,
        help="diffusers model name",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="k_euler",
        help="sampler to use (ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_lms)"
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
        default=0,
        help="random seed for sampling (0 for auto, auto-range defined in DEFAULT_SAMPLE_SETTINGS)",
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
        default="",
        help="attach an id that can be used for identification or searching purposes later",
    )
    
    return parser
    
def get_default_args():
    return get_args_parser().parse_args()
    
def build_grpc_request_dict(args):
    global DEFAULT_SAMPLE_SETTINGS
    
    # auto-seed if none provided
    if args.seed != 0:
        seed = args.seed
    else:
        args.auto_seed =  int(np.random.randint(DEFAULT_SAMPLE_SETTINGS.auto_seed_range[0], DEFAULT_SAMPLE_SETTINGS.auto_seed_range[1]))
        seed = args.auto_seed
        
    return {
        "height": args.h,
        "width": args.w,
        "start_schedule": None, #args.start_schedule,
        "end_schedule": None,   #args.end_schedule,
        "cfg_scale": args.scale,
        "eta": 0.,#args.eta,
        "sampler": grpc_client.get_sampler_from_str(args.sampler),
        "steps": args.steps,
        "seed": seed,
        "samples": args.n,
        "init_image": None, #args.init_image,
        "mask_image": None, #args.mask_image,
        #"negative_prompt": args.negative_prompt
    }    