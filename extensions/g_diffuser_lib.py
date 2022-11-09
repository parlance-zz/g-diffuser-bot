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


g_diffuser_lib.py - core diffuser / grpc client operations and lib utilities

"""

from g_diffuser_config import DEFAULT_PATHS, GRPC_SERVER_SETTINGS, CLI_SETTINGS
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS

import os

if os.name == "nt": # this kludge can help make long file paths on windows more reliable
    import ntpath
    ntpath.realpath = ntpath.abspath 
else:
    import readline # required on linux for functional arrow keys in the python interactive interpreter =\

import datetime
import argparse
import uuid
import pathlib
import json
import re
import subprocess
import glob
import socket
import aiofiles

import numpy as np
import cv2

from extensions import grpc_client
from extensions import g_diffuser_utilities as gdl_utils

from torch import autocast

global GRPC_SERVER_PROCESS
GRPC_SERVER_PROCESS = None

global SUPPORTED_SAMPLERS_LIST
SUPPORTED_SAMPLERS_LIST = list(grpc_client.SAMPLERS.keys())

def run_string(run_string, cwd=".", log_path=None, err_path=None):  # run shell command asynchronously, return subprocess
    print(run_string + " (cwd="+str(cwd)+")")

    if log_path: log_file = open(log_path, "w", 1)
    else: log_file = None
    if err_path: err_file = open(err_path, "w", 1)
    else: err_file = None

    #if log_path == None: log_file = subprocess.DEVNULL
    #if err_path == None: err_file = subprocess.DEVNULL

    process = subprocess.Popen(run_string, shell=True, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=err_file, encoding='ascii')

    assert(process)
    return process

def get_socket_listening_status(host_str):
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        ip = socket.gethostbyname(host_str.split(":")[0])
        port = int(host_str.split(":")[1])
        _socket.connect((ip, port)); _socket.shutdown(socket.SHUT_RDWR)
        return True
    except:
        return False

def validate_resolution(width, height, init_image_dims):  # clip output dimensions at max_resolution, while keeping the correct resolution granularity,
                                                          # while roughly preserving aspect ratio. if width or height are None they are taken from the init_image
    global DEFAULT_SAMPLE_SETTINGS

    if init_image_dims == None:
        if not width: width = DEFAULT_SAMPLE_SETTINGS.resolution[0]
        if not height: height = DEFAULT_SAMPLE_SETTINGS.resolution[1]
    else:
        if not width: width = init_image_dims[0]
        if not height: height = init_image_dims[1]
        
    aspect_ratio = width / height 
    if width > DEFAULT_SAMPLE_SETTINGS.max_resolution[0]:
        width = DEFAULT_SAMPLE_SETTINGS.max_resolution[0]
        height = int(width / aspect_ratio + .5)
    if height > DEFAULT_SAMPLE_SETTINGS.max_resolution[1]:
        height = DEFAULT_SAMPLE_SETTINGS.max_resolution[1]
        width = int(height * aspect_ratio + .5)
        
    width = int(width / float(DEFAULT_SAMPLE_SETTINGS.resolution_granularity) + 0.5) * DEFAULT_SAMPLE_SETTINGS.resolution_granularity
    height = int(height / float(DEFAULT_SAMPLE_SETTINGS.resolution_granularity) + 0.5) * DEFAULT_SAMPLE_SETTINGS.resolution_granularity
    width = np.clip(width, DEFAULT_SAMPLE_SETTINGS.resolution_granularity, DEFAULT_SAMPLE_SETTINGS.max_resolution[0])
    height = np.clip(height, DEFAULT_SAMPLE_SETTINGS.resolution_granularity, DEFAULT_SAMPLE_SETTINGS.max_resolution[1])

    return int(width), int(height)
    
def get_random_string(digits=8):
    uuid_str = str(uuid.uuid4())
    return uuid_str[0:digits] # shorten uuid, don't need that many digits usually

def print_namespace(namespace, debug=False, verbosity_level=0, indent=4):
    namespace_dict = vars(strip_args(namespace, level=verbosity_level))
    if debug:
        for arg in namespace_dict:
            print(arg+"='"+str(namespace_dict[arg]) + "' "+str(type(namespace_dict[arg])))
    else:
        print(json.dumps(namespace_dict, indent=indent))
    return

def get_default_output_name(name, truncate_length=100, force_ascii=False):
    if force_ascii: name = str(name.encode('utf-8').decode('ascii', 'ignore'))
    sanitized_name = re.sub(r'[\\/*?:"<>|]',"", name).replace(".","").replace("'","").replace("\t"," ").replace(" ","_").strip()
    if (truncate_length > len(sanitized_name)) or (truncate_length==0): truncate_length = len(sanitized_name)
    if truncate_length < len(sanitized_name):  sanitized_name = sanitized_name[0:truncate_length]
    return sanitized_name

def get_noclobber_checked_path(base_path, file_path):
    clobber_num_padding = 3
    full_path = base_path+"/"+file_path
    file_path_noext, file_path_ext = os.path.splitext(file_path)
    existing_count = len(glob.glob(base_path+"/"+file_path_noext+"*"+file_path_ext))
    return file_path_noext+"_x"+str(existing_count).zfill(clobber_num_padding)+file_path_ext

def save_image(cv2_image, file_path):
    assert(file_path); (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    cv2.imwrite(file_path, cv2_image)
    return

async def save_image_async(cv2_image, file_path):
    assert(file_path); (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    img_bytes = np.array(cv2.imencode(os.path.splitext(file_path)[1], cv2_image)[1]).tobytes()
    async with aiofiles.open(file_path, mode="wb") as out_file:
        await out_file.write(img_bytes)
        await out_file.close()
    return

def save_json(_dict, file_path):
    assert(file_path); (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    with open(file_path, "w") as file:
        json.dump(_dict, file, indent=4)
        file.close()
    return
    
async def save_json_async(_dict, file_path):
    assert(file_path); (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    async with aiofiles.open(file_path, "w") as file:
        await file.write(json.dumps(_dict, indent=4))
        await file.close()
    return

def load_json(file_path):
    assert(file_path); (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    with open(file_path, "r") as file:
        data = json.load(file)
        file.close()
    return data
    
def strip_args(args, level=0): # remove args we wouldn't want to print or serialize, higher levels strip additional irrelevant fields
    args_stripped = argparse.Namespace(**(vars(args).copy()))

    if level >=1: # keep just the basics for most printing
        if "debug" in args_stripped:
            if not args_stripped.debug: del args_stripped.debug

        if "seed" in args_stripped:
            if args_stripped.seed == 0: del args_stripped.seed

        if "command" in args_stripped: del args_stripped.command
        if "interactive" in args_stripped: del args_stripped.interactive
        if "load_args" in args_stripped: del args_stripped.load_args
        
        if "init_time" in args_stripped: del args_stripped.init_time
        if "start_time" in args_stripped: del args_stripped.start_time
        if "end_time" in args_stripped: del args_stripped.end_time
        if "elapsed_time" in args_stripped: del args_stripped.elapsed_time

        if "output_path" in args_stripped: del args_stripped.output_path
        if "final_output_path" in args_stripped: del args_stripped.final_output_path
        if "output_name" in args_stripped: del args_stripped.output_name
        if "final_output_name" in args_stripped: del args_stripped.final_output_name
        if "output_file" in args_stripped: del args_stripped.output_file
        if "output_file_type" in args_stripped: del args_stripped.output_file_type
        if "grid_image" in args_stripped: del args_stripped.grid_image
        if "args_file" in args_stripped: del args_stripped.args_file
        if "no_json" in args_stripped: del args_stripped.no_json

        if "uuid_str" in args_stripped: del args_stripped.uuid_str
        if "status" in args_stripped: del args_stripped.status
        if "err_txt" in args_stripped: del args_stripped.err_txt

        if "noise_q" in args_stripped: del args_stripped.noise_q
        if "noise_end" in args_stripped: del args_stripped.noise_end
        if "noise_eta" in args_stripped: del args_stripped.noise_eta

        if "init_img" in args_stripped:
            if args_stripped.init_img == "": # if there was no input image these fields are not relevant
                del args_stripped.init_img
                if "noise_start" in args_stripped: del args_stripped.noise_start

        if "model_name" in args_stripped:
            if "clip" not in args_stripped.model_name:
                if "guidance_strength" in args_stripped: del args_stripped.guidance_strength


    return args_stripped
    
def get_grid_layout(num_samples):
    def factorize(num):
        return [n for n in range(1, num + 1) if num % n == 0]
    factors = factorize(num_samples)
    median_factor = factors[len(factors)//2]
    rows = median_factor
    columns = num_samples // rows
    return (columns, rows)
    
def get_image_grid(imgs, layout, mode="columns"): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    width, height = (imgs[0].shape[0], imgs[0].shape[1])

    np_grid = np.zeros((layout[0]*width, layout[1]*height, 3), dtype="uint8")
    for i, img in enumerate(imgs):
        if mode != "rows":
            paste_x = i // layout[1] * width
            paste_y = i % layout[1] * height
        else:
            paste_x = i % layout[0] * width
            paste_y = i // layout[0] * height
        np_grid[paste_x:paste_x+width, paste_y:paste_y+height, :] = img[:]

    return np_grid

def get_annotated_image(image, args):
    if "annotation" not in args: return image
    if not args.annotation: return image

    annotation_font = cv2.FONT_HERSHEY_SIMPLEX
    annotation_position = (8, 31) # todo: hard-coding top-left with small offset for now
    annotation_scale = 7/8.
    annotation_linetype = cv2.LINE_8 | cv2.LINE_AA
    annotation_linethickness = 2
    annotation_outline_radius = 4

    image_copy = image.copy()
    try:
        annotation_color = (0,0,0)
        for x in range(-annotation_outline_radius, annotation_outline_radius+1, annotation_linethickness):
            for y in range(-annotation_outline_radius, annotation_outline_radius+1, annotation_linethickness):
                cv2.putText(image_copy, args.annotation, 
                    (annotation_position[0]+x, annotation_position[1]+y),
                    annotation_font, 
                    annotation_scale,
                    (0,0,0),
                    annotation_linethickness,
                    annotation_linetype)
        annotation_color = (175,175,175)
        cv2.putText(image_copy, args.annotation, 
            annotation_position,
            annotation_font, 
            annotation_scale,
            (255,255,255),
            annotation_linethickness,
            annotation_linetype)
    except Exception as e:
        print("Error annotating sample - " + str(e))
        return image
    return image_copy

def load_image(args):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS
    assert(DEFAULT_PATHS.inputs)
    MASK_CUTOFF_THRESHOLD = 200. # this will force the image mask to 0 if opacity falls below a threshold. set to 255. to disable

    final_init_img_path = (pathlib.Path(DEFAULT_PATHS.inputs) / args.init_img).as_posix()
    
    # load and resize input image to multiple of 8x8
    init_image = cv2.imread(final_init_img_path, cv2.IMREAD_UNCHANGED)
    init_image_dims = (init_image.shape[1], init_image.shape[0])
    width, height = validate_resolution(args.w, args.h, init_image_dims)
    if (width, height) != (init_image.shape[1], init_image.shape[0]):
        if args.debug: print("Resizing input image to (" + str(width) + ", " + str(height) + ")")
        init_image = cv2.resize(init_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    args.w = width
    args.h = height
    
    num_channels = init_image.shape[2]
    if num_channels == 4:     # input image has an alpha channel, setup mask for in/out-painting
        args.noise_start = 1. # override img2img "strength", for in/out-painting this should always be maxed
        mask_image = 255. - init_image[:,:,3] # extract mask from alpha channel and invert
        init_image = 0. + init_image[:,:,0:3]      # strip mask from init_img / convert to rgb

        # apply mask cutoff threshold, this is required because bad paint programs corrupt color data by premultiplying alpha
        # using the 8-bit opacity mask, resulting in unexpected artifacts in areas that are almost but not completely erased
        mask_image = mask_image*(mask_image < MASK_CUTOFF_THRESHOLD) + (mask_image >= MASK_CUTOFF_THRESHOLD)*255.
        init_image *= gdl_utils.np_img_grey_to_rgb(mask_image) < 255. # force color data in erased areas to 0

        if args.sampler == "k_euler":
            print("Warning: k_euler is not currently supported for in-painting, switching to sampler=k_euler_ancestral")
            args.sampler = "k_euler_ancestral" # k_euler currently does not add noise during sampling
        
    elif num_channels == 3: # rgb image, regular img2img without a mask
        mask_image = None
    else:
        raise Exception("Error loading init_image "+final_init_img_path+": unsupported image format")

    return init_image, mask_image

def build_sample_args(args):
    global DEFAULT_SAMPLE_SETTINGS
    if not args.output_name: args.final_output_name = get_default_output_name(args.prompt)
    else: args.final_output_name = args.output_name
    if not args.output_path: args.final_output_path = args.final_output_name
    else: args.final_output_path = args.output_path
        
    if not args.seed: # no seed provided
        if not ("auto_seed" in args): # no existing auto-seed
            args.auto_seed = int(np.random.randint(DEFAULT_SAMPLE_SETTINGS.auto_seed_range[0], DEFAULT_SAMPLE_SETTINGS.auto_seed_range[1])) # new random auto-seed
    else:
        if ("auto_seed" in args): del args.auto_seed # if a seed is provided just strip out the auto_seed entirely

    if args.init_img != "": # load input image if we have one
        init_image, mask_image = load_image(args)
    else:
        init_image, mask_image = (None, None)
        if not args.w: args.w = DEFAULT_SAMPLE_SETTINGS.resolution[0] # if we don't have an input image, it's size can't be used as the default resolution
        if not args.h: args.h = DEFAULT_SAMPLE_SETTINGS.resolution[1]

    if "grid_image" in args: del args.grid_image

    return init_image, mask_image

def get_samples(args, write=True, no_grid=False):
    global DEFAULT_PATHS, GRPC_SERVER_SETTINGS
    assert((args.n > 0) or write) # repeating forever without writing to disk wouldn't make much sense

    try:
        init_image, mask_image = build_sample_args(args)
    except Exception as e:
        args.status = -1; args.err_txt = str(e) # error status
        if args.debug: raise
        else: print("Error: " + args.err_txt)
        return [],[]

    samples = []; sample_files = []
    stability_api = grpc_client.StabilityInference("localhost:50051", None, engine=args.model_name, verbose=False)
    while True: # watch out! a wild shrew!
        try:
            request_dict = build_grpc_request_dict(args, init_image, mask_image)
            answers = stability_api.generate(**request_dict)
            grpc_samples = grpc_client.process_artifacts_from_answers("", answers, write=False, verbose=False)

            start_time = datetime.datetime.now(); args.start_time = str(start_time)
            for path, artifact in grpc_samples:
                end_time = datetime.datetime.now(); args.end_time = str(end_time); args.elapsed_time = str(end_time-start_time)
                args.status = 2; args.err_txt = "" # completed successfully

                image = cv2.imdecode(np.fromstring(artifact.binary, dtype="uint8"), cv2.IMREAD_UNCHANGED)
                if "annotation" in args: image = get_annotated_image(image, args)
                samples.append(image)

                if write:
                    args.uuid_str = get_random_string(digits=16) # new uuid for new sample
                    sample_files.append(save_sample(image, args))

                if args.seed: args.seed += 1 # increment seed or random seed if none was given as we go through the batch
                else: args.auto_seed += 1
                if (len(samples) < args.n) or (args.n <= 0): # reset start time for next sample if we still have samples left
                    start_time = datetime.datetime.now(); args.start_time = str(start_time)

            if args.n > 0: break # if we had a set number of samples then we are done

        except Exception as e:
            args.status = -1; args.err_txt = str(e) # error status
            if args.debug: raise
            else: print("Error: " + args.err_txt)
            return samples, sample_files

    if write and len(samples) > 1 and (not no_grid):
        save_samples_grid(samples, args) # if batch size > 1 and write to disk is enabled, save composite "grid image"
    return samples, sample_files

def save_sample(sample, args):
    global DEFAULT_PATHS, CLI_SETTINGS
    assert(DEFAULT_PATHS.outputs)
    if args.seed: seed = args.seed
    else: seed = args.auto_seed

    seed_num_padding = 5
    filename = args.final_output_name+"_s"+str(seed).zfill(seed_num_padding)+".png"
    args.output_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, args.final_output_path+"/"+filename)
    args.output_file_type = "img" # the future is coming, hold on to your butts

    final_path = DEFAULT_PATHS.outputs+"/"+args.output_file
    save_image(sample, final_path)
    print("Saved " + final_path)

    if not args.no_json:
        args.args_file = args.final_output_path+"/json/"+args.final_output_name+"_s"+str(seed).zfill(seed_num_padding)+".json"
        args.args_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, args.args_file) # add suffix if filename already exists
        save_json(vars(strip_args(args)), DEFAULT_PATHS.outputs+"/"+args.args_file)
    return args.output_file

def save_samples_grid(samples, args):
    global CLI_SETTINGS
    assert(len(samples)> 1)
    grid_layout = get_grid_layout(len(samples))
    grid_image = get_image_grid(samples, grid_layout)
    filename = "grid_" + args.final_output_name + ".jpg"
    output_file = args.final_output_path+"/" + filename
    output_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, output_file)

    final_path = DEFAULT_PATHS.outputs+"/"+output_file
    save_image(grid_image, final_path)
    print("Saved grid " + final_path)
    args.output_file = output_file
    return args.output_file

def start_grpc_server(args):
    global DEFAULT_PATHS, GRPC_SERVER_SETTINGS, GRPC_SERVER_PROCESS, CLI_SETTINGS
    if args.debug: load_start_time = datetime.datetime.now()

    host = "localhost:50051"
    if get_socket_listening_status(host):
        print("Found running GRPC server listening on " + host)
        return None
    else:
        print("Starting GRPC server...")

    #grpc_server_run_string = "docker run --pull always --gpus all -p 50051:50051 -e HF_API_TOKEN="+GRPC_SERVER_SETTINGS.hf_token
    grpc_server_run_string = "docker run --gpus all -p 50051:50051 -e HF_API_TOKEN="+GRPC_SERVER_SETTINGS.hf_token
    if GRPC_SERVER_SETTINGS.enable_local_network_access: grpc_server_run_string += " -e SD_LISTEN_TO_ALL=1"
    if GRPC_SERVER_SETTINGS.enable_mps: grpc_server_run_string += " -e SD_ENABLE_MPS=1"
    if GRPC_SERVER_SETTINGS.refresh_models: grpc_server_run_string += " -e SD_REFRESH_MODELS=*"
    grpc_server_run_string += " -e SD_ENGINECFG=/weights/models.yaml -e SD_NSFW_BEHAVIOUR="+GRPC_SERVER_SETTINGS.nsfw_behaviour
    grpc_server_run_string += " -e SD_VRAM_OPTIMISATION_LEVEL="+str(GRPC_SERVER_SETTINGS.memory_optimization_level)
    grpc_server_run_string += ' -v "'+DEFAULT_PATHS.models+'":/huggingface -v "'+DEFAULT_PATHS.models+'":/weights'
    if DEFAULT_PATHS.sdgrpcserver_docker_volume: grpc_server_run_string += ' -v "'+DEFAULT_PATHS.sdgrpcserver_docker_volume+'":/sdgrpcserver'
    grpc_server_run_string += " " + GRPC_SERVER_SETTINGS.docker_image_name
    
    GRPC_SERVER_PROCESS = run_string(grpc_server_run_string)
    if args.debug: print("sd_grpc_server start time : " + str(datetime.datetime.now() - load_start_time))
    return GRPC_SERVER_PROCESS

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
        help="random starting seed for sampling (0 for random)",
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
        help="classifier-free guidance scale (~amount of change per step)",
    )
    parser.add_argument(
        "--noise_q",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.noise_q,
        help="falloff of shaped noise distribution for in/out-painting ( > 0), 1 is matched, lower values mean smaller features and higher means larger features",
    )
    parser.add_argument(
        "--noise_start",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.noise_start,
        help="formerly known as strength, this is the overall amount of change for img2img",
    )
    parser.add_argument(
        "--noise_end",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.noise_end,
        help="this param affects the sampler noise schedule",
    )
    parser.add_argument(
        "--noise_eta",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.noise_eta,
        help="this param affects the sampler noise schedule",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_SAMPLE_SETTINGS.negative_prompt,
        help="the 'negative' prompt guides generation the same way prompt does, but with a weight of -1.",
    )
    parser.add_argument(
        "--guidance_strength",
        type=float,
        default=DEFAULT_SAMPLE_SETTINGS.guidance_strength,
        help="this param controls 'clip guided' generation models only",
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
        "--init-img",
        type=str,
        default="",
        help="path to the input image",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to store output samples (relative to root outputs path, by default this uses the prompt)",
        default="",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="use a specified output file name instead of one based on the prompt",
        default="",
    )
    """
    parser.add_argument(
        "--show",
        action='store_true',
        default=False,
        help="show the output after sample is completed",
    )  
    """  
    parser.add_argument(
        "--interactive",
        action='store_true',
        default=False,
        help="enters an interactive command line mode to generate multiple samples",
    )
    parser.add_argument(
        "--load-args",
        type=str,
        default="",
        help="load and use a saved set of arguments from ./inputs/json",
    )
    parser.add_argument(
        "--no-json",
        action='store_true',
        default=False,
        help="disable saving arg files for each sample output in ./outputs/output_path/json",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="enable verbose CLI output and debug file dumps",
    )
    return parser
    
def get_default_args():
    return get_args_parser().parse_args([])
    
def build_grpc_request_dict(args, init_image, mask_image):
    global DEFAULT_SAMPLE_SETTINGS
    # use auto-seed if none provided
    if args.seed: seed = args.seed
    else:
        args.auto_seed
        seed = args.auto_seed
    
    if init_image is None: init_image_bytes = None
    else: init_image_bytes = np.array(cv2.imencode(".png", init_image)[1]).tobytes()
    if mask_image is None: mask_image_bytes = None
    else: mask_image_bytes = np.array(cv2.imencode(".png", mask_image)[1]).tobytes()

    # if repeating just use a giant batch size for now
    if args.n <= 0: n = int(1e10)
    else: n = args.n

    return {
        "prompt": args.prompt,
        "height": args.h,
        "width": args.w,
        "start_schedule": args.noise_start,
        "end_schedule": args.noise_end,
        "cfg_scale": args.scale,
        "eta": args.noise_eta,
        "sampler": grpc_client.get_sampler_from_str(args.sampler),
        "steps": args.steps,
        "seed": seed,
        "samples": n,
        "init_image": init_image_bytes,
        "mask_image": mask_image_bytes,
        "negative_prompt": args.negative_prompt,
        "guidance_preset": grpc_client.generation.GUIDANCE_PRESET_SIMPLE if args.guidance_strength > 0. else grpc_client.generation.GUIDANCE_PRESET_NONE,
        "guidance_strength": args.guidance_strength,
        "guidance_prompt": args.prompt,   
    }    