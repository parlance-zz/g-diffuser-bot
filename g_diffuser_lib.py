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
import importlib

import numpy as np
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

import torch
from torch import autocast

from extensions.stable_diffusion_grpcserver.sdgrpcserver import server

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
    if "loaded_pipes" in args_stripped: del args_stripped.loaded_pipes
    
    if level >=1: # keep just the basics for most printing
        if "command" in args_stripped: del args_stripped.command
        if "seed" in args_stripped: del args_stripped.seed
        if "use_optimized" in args_stripped: del args_stripped.use_optimized
        if "debug" in args_stripped: del args_stripped.debug
        if "interactive" in args_stripped: del args_stripped.interactive
        if "load_args" in args_stripped: del args_stripped.load_args
        if "no_json" in args_stripped: del args_stripped.no_json
        if "pipe_list" in args_stripped: del args_stripped.pipe_list
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
    
def get_image_grid(imgs, layout): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(layout[1]*w, layout[0]*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%layout[1]*w, i//layout[1]*h))
    return grid

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
    global DEFAULT_SAMPLE_SETTINGS
    global MODEL_DEFAULTS, LOADED_MODEL_ARGS
    
    if args.debug:
        importlib.reload(ext) # this allows us to test modifications to extensions without reloading the cli or model
    
    args.uuid_str = get_random_string(digits=16) # new uuid for new sample(s)
    args.status = 1 # running
    
    if args.init_img != "": # load input image if we have one
        init_image, mask_image = load_image(args)
    else:
        init_image, mask_image = (None, None)
        if not args.w: args.w = DEFAULT_SAMPLE_SETTINGS.resolution[0] # if we don't have an input image, it's size can't be used as the default resolution
        if not args.h: args.h = DEFAULT_SAMPLE_SETTINGS.resolution[1]
        
    # check if args came with model name or load-time settings we can't accommodate, at least until we can load models on the fly. if mismatch replace with the loaded model args
    if args.model_name != LOADED_MODEL_ARGS.model_name:
        if args.debug:
            print("Warning - model name requested in args '" + args.model_name + "' is not loaded")
            print("Setting args.model_name to loaded model '" + LOADED_MODEL_ARGS.model_name + "'...")
        args.model_name = LOADED_MODEL_ARGS.model_name  
    if args.use_optimized != LOADED_MODEL_ARGS.use_optimized: # check for memory optimizations requested in args mismatch against loaded model settings
        if args.debug: print("Warning - use_optimized requested in args (" + str(args.use_optimized) + ") does not match loaded model settings (" + str(LOADED_MODEL_ARGS.use_optimized) + ")")
        args.use_optimized = LOADED_MODEL_ARGS.use_optimized

    assert(len(LOADED_MODEL_ARGS.pipe_list) > 0) # we need to have loaded at least one pipe
    pipe_name = LOADED_MODEL_ARGS.pipe_list[0]
    
    start_time = datetime.datetime.now()
    args.start_time = str(start_time)
    args.used_pipe = pipe_name
    error_in_sampling = False
    samples = []
    
    with autocast("cuda"):
    
        if args.debug: print("Using " + pipe_name + " pipeline...")
        pipe = args.loaded_pipes[pipe_name]
        assert(pipe)
        
        for n in range(args.n): # batched mode doesn't seem to accomplish much besides using more memory
            if args.status == 3: return # if command is cancelled just bail out asap
            try:
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
            except Exception as e:
                print("Error running pipeline '" + pipe_name + "' - " + str(e))
                raise
                sample = None
                error_in_sampling = True # set args.status after all samples are completed only, otherwise stay "running"
                
            if sample: samples.append(sample["sample"][0])
    
    end_time = datetime.datetime.now()
    args.end_time = str(end_time)
    args.elapsed_time = str(end_time-start_time)
    
    if (len(samples) == 0) or error_in_sampling:    
        args.status = -1 # error running command
    else:
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
    
def load_pipelines(args):
    global DEFAULT_PATHS
    global MODEL_DEFAULTS
    global LOADED_MODEL_ARGS
    
    pipe_map = { "g-diffuser-lib_super": ext.G_Diffuser_SuperPipeline }
    pipe_list = ["g-diffuser-lib_super"]
    
    hf_token = None
    if "hf_token" in MODEL_DEFAULTS: hf_token = MODEL_DEFAULTS.hf_token
    if "hf_token" in args: hf_token = args.hf_token
    args.hf_token = hf_token
    
    use_optimized = MODEL_DEFAULTS.use_optimized
    if args.use_optimized: use_optimized = args.use_optimized
    if args.model_name: model_name = args.model_name
    else: model_name = MODEL_DEFAULTS.model_name
    args.model_name = model_name
    
    LOADED_MODEL_ARGS = argparse.Namespace()  # remember loaded model settings, this is needed for now until we have the memory and/or logistics to efficiently load models on the fly
    LOADED_MODEL_ARGS.model_name = model_name
    LOADED_MODEL_ARGS.use_optimized = use_optimized
    LOADED_MODEL_ARGS.pipe_list = pipe_list
    
    if not hf_token: final_model_path = (pathlib.Path(DEFAULT_PATHS.models) / model_name).as_posix() # not using hf token
    else: final_model_path = model_name
    print("Using model: " + final_model_path)
    
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
            final_model_path, 
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
        )
        pipe = pipe.to("cuda")
        #setattr(pipe, "safety_checker", dummy_checker)
        if use_optimized == True:
            pipe.enable_attention_slicing() # use attention slicing in optimized mode
            
        loaded_pipes[pipe_name] = pipe
        if args.debug: print_namespace(pipe, debug=1)
    if args.debug: print("load pipelines time : " + str(datetime.datetime.now() - load_start_time))
    args.loaded_pipes = loaded_pipes
    
    return args.loaded_pipes

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
    """
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_DEFAULTS.model_name,
        help="path to downloaded diffusers model (relative to default models path), or name of model if using a huggingface token",
    )
    parser.add_argument(
        "--use-optimized",
        action='store_true',
        default=False,
        help="enable memory optimizations that are currently available in diffusers",
    )
    """
    
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

server.main(enginecfg=DEFAULT_PATHS.model_cfg, models_root=DEFAULT_PATHS.models)