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

import os

if os.name == "nt": # this kludge can help make long file paths on windows more reliable
    import ntpath
    ntpath.realpath = ntpath.abspath 
else:
    import readline # required on linux for functional arrow keys in the python interactive interpreter =\

import sys
import datetime
import argparse
from argparse import Namespace
import uuid
import pathlib
import re
import glob
import socket
import asyncio
import threading
from threading import Thread
from dotenv import load_dotenv

import yaml
from yaml import CLoader as Loader

import numpy as np
import cv2

from modules import sdgrpcserver_client as grpc_client
from modules import g_diffuser_utilities as gdl_utils

GRPC_SERVER_SUPPORTED_SAMPLERS_LIST = list(grpc_client.SAMPLERS.keys())
GRPC_SERVER_ENGINE_STATUS = []
GRPC_SERVER_LOCK = asyncio.Lock()

class SimpleLogger(object):
    def __init__(self, log_path, mode="w"):
        try: self.log = open(log_path, mode)
        except: return
        # hijack stdout, stdin, stderr
        sys.stdout = self
        sys.stdin = self
        sys.stderr = self
        return
    def __del__(self):
        # restore original stream values
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
        sys.stderr = sys.__stderr__
        if self.log: self.log.close()
    def write(self, data):
        self.log.write(data)
        self.log.flush()
        sys.__stdout__.write(data)
        sys.__stdout__.flush()
    def readline(self):
        s = sys.__stdin__.readline()
        sys.__stdin__.flush()
        self.log.write(s)
        self.log.flush()
        return s
    def flush(foo):
        return

def start_grpc_server():
    global GRPC_SERVER_SETTINGS, DEFAULT_SAMPLE_SETTINGS
    if get_socket_listening_status(GRPC_SERVER_SETTINGS.host):
        print("\nFound running SDGRPC server listening on {0}".format(GRPC_SERVER_SETTINGS.host))
    else:
        raise Exception("Could not connect to SDGRPC server at {0}, is the server running?".format(GRPC_SERVER_SETTINGS.host))

    try:
        models = show_models()
        all_models_ready = True
        model_ids = {}
        for model in models:
            all_models_ready &= model["ready"]
            model_ids[model["id"]] = True
        if not all_models_ready:
            print("Not all models are ready yet, the server may still be starting up...")

        if not model_ids.get(DEFAULT_SAMPLE_SETTINGS.model_name, False): # check that the selected default model exists on the server
            new_default_model_id = models[0]["id"]
            print ("Warning: Default model {0} is not available on server, using first available model {1} instead.".format(DEFAULT_SAMPLE_SETTINGS.model_name, new_default_model_id))
            DEFAULT_SAMPLE_SETTINGS.model_name = new_default_model_id

    except Exception as e:
        raise Exception("Unable to query server status at {0} with key'{1}', is the host/key correct?".format(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.grpc_key))

    return models

def get_socket_listening_status(host_str):
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        ip = socket.gethostbyname(host_str.split(":")[0])
        port = int(host_str.split(":")[1])
        _socket.connect((ip, port))
        _socket.shutdown(socket.SHUT_RDWR)
        return True
    except:
        return False

def get_models():
    global GRPC_SERVER_SETTINGS, GRPC_SERVER_ENGINE_STATUS
    stability_api = grpc_client.StabilityInference(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.grpc_key, wait_for_ready=False)
    try:
        GRPC_SERVER_ENGINE_STATUS = stability_api.list_engines()
    except Exception as e:
        print("Error: Could not query SDGRPC server - {0}".format(e))
        return None
    return GRPC_SERVER_ENGINE_STATUS

def show_models(models=None):
    if models == None: models = get_models()
    if models == None: return None
    print("Found {0} model(s) on the SDGRPC server:".format(len(models)))
    print(yaml.dump(models, sort_keys=False))
    return models
        
def load_config():
    global DEFAULT_PATHS
    global GRPC_SERVER_SETTINGS
    global DISCORD_BOT_SETTINGS
    global DEFAULT_SAMPLE_SETTINGS

    # this env is already loaded if this is a sub-process of the server, but if it isn't we need to load it here
    load_dotenv("config.ini")

    # default paths and main settings are located in the config file and passed as environment variables
    DEFAULT_PATHS = argparse.Namespace()
    DEFAULT_PATHS.root = str(os.environ.get("GDIFFUSER_PATH_ROOT", "g-diffuser"))
    DEFAULT_PATHS.inputs = str(os.environ.get("GDIFFUSER_PATH_INPUTS", "inputs"))
    DEFAULT_PATHS.outputs = str(os.environ.get("GDIFFUSER_OUTPUTS_PATH", "outputs"))
    DEFAULT_PATHS.temp = str(os.environ.get("GDIFFUSER_TEMP_PATH", "temp"))
    DEFAULT_PATHS.bot = str(os.environ.get("GDIFFUSER_BOT_PATH", "bot"))
    DEFAULT_PATHS.defaults_file = str(os.environ.get("GDIFFUSER_DEFAULTS_FILE", "defaults.yaml"))

    GRPC_SERVER_SETTINGS = argparse.Namespace()
    GRPC_SERVER_SETTINGS.host = str(os.environ.get("SD_GRPC_HOST", "localhost"))
    GRPC_SERVER_SETTINGS.grpc_port = int(os.environ.get("SD_GRPC_PORT", "50051"))
    GRPC_SERVER_SETTINGS.grpc_key = str(os.environ.get("SD_GRPC_KEY", ""))
    GRPC_SERVER_SETTINGS.grpc_web = bool(int(os.environ.get("SD_GRPC_WEB", "0")))
    GRPC_SERVER_SETTINGS.nsfw_behavior = os.environ.get("SD_NSFW_BEHAVIOUR", "block")
    GRPC_SERVER_SETTINGS.vram_optimization_level = int(os.environ.get("SD_VRAM_OPTIMISATION_LEVEL", "2"))
    if GRPC_SERVER_SETTINGS.host == "localhost":
        GRPC_SERVER_SETTINGS.grpc_web = False
        GRPC_SERVER_SETTINGS.host = "localhost:{0}".format(GRPC_SERVER_SETTINGS.grpc_port)

    DISCORD_BOT_SETTINGS = argparse.Namespace()
    DISCORD_BOT_SETTINGS.token = str(os.environ.get("DISCORD_BOT_TOKEN", ""))
    DISCORD_BOT_SETTINGS.state_file_path = str(os.environ.get("DISCORD_BOT_STATE_FILE", "g_diffuser_bot.json"))
    DISCORD_BOT_SETTINGS.default_output_n = int(os.environ.get("DISCORD_BOT_DEFAULT_OUTPUT_N", "1"))
    DISCORD_BOT_SETTINGS.max_output_limit = int(os.environ.get("DISCORD_BOT_MAX_OUTPUT_LIMIT", "3"))
    DISCORD_BOT_SETTINGS.max_steps_limit = int(os.environ.get("DISCORD_BOT_MAX_STEPS_LIMIT", "100"))

    # try loading sampling defaults from the defaults.yaml file
    try:
        defaults = load_yaml(DEFAULT_PATHS.defaults_file)
    except Exception as e:
        print("Warning: Could not load defaults file {0} - {1}".format(DEFAULT_PATHS.defaults_file, e))
        defaults = {}

    DEFAULT_SAMPLE_SETTINGS = argparse.Namespace()
    DEFAULT_SAMPLE_SETTINGS.prompt = ""
    DEFAULT_SAMPLE_SETTINGS.seed = 0 # 0 means use auto seed    
    DEFAULT_SAMPLE_SETTINGS.model_name = str(defaults.get("model_name", "default"))
    DEFAULT_SAMPLE_SETTINGS.num_samples = int(defaults.get("num_samples", 1))
    DEFAULT_SAMPLE_SETTINGS.sampler = str(defaults.get("sampler", "dpmspp_2"))
    DEFAULT_SAMPLE_SETTINGS.steps = int(defaults.get("steps", 50))
    #DEFAULT_SAMPLE_SETTINGS.max_steps = int(defaults.get("max_steps", 150))
    DEFAULT_SAMPLE_SETTINGS.cfg_scale = float(defaults.get("cfg_scale", 14.))
    DEFAULT_SAMPLE_SETTINGS.guidance_strength = float(defaults.get("guidance_strength", 0.5))
    DEFAULT_SAMPLE_SETTINGS.negative_prompt = str(defaults.get("negative_prompt", ""))

    DEFAULT_SAMPLE_SETTINGS.width = int(defaults.get("default_resolution", {}).get("width", 512))
    DEFAULT_SAMPLE_SETTINGS.height = int(defaults.get("default_resolution", {}).get("height", 512))
    DEFAULT_SAMPLE_SETTINGS.max_width = int(defaults.get("max_resolution", {}).get("width", 960))
    DEFAULT_SAMPLE_SETTINGS.max_height = int(defaults.get("max_resolution", {}).get("height", 960))

    DEFAULT_SAMPLE_SETTINGS.hires_fix = bool(defaults.get("hires_fix", False))
    DEFAULT_SAMPLE_SETTINGS.seamless_tiling = bool(defaults.get("seamless_tiling", False))

    DEFAULT_SAMPLE_SETTINGS.init_image = "" # used to supply an input image for in/out-painting or img2img
    DEFAULT_SAMPLE_SETTINGS.img2img_strength = float(defaults.get("img2img_strength", 0.65))

    DEFAULT_SAMPLE_SETTINGS.auto_seed = 0 # if auto_seed is 0, at sampling time this is replaced with a random seed
                                          # from the auto_seed_range. if auto_seed is not 0 it is incremented by 1 instead
    DEFAULT_SAMPLE_SETTINGS.auto_seed_low = int(defaults.get("auto_seed_range", {}).get("low", 10000))
    DEFAULT_SAMPLE_SETTINGS.auto_seed_high = int(defaults.get("auto_seed_range", {}).get("high", 99999))

    DEFAULT_SAMPLE_SETTINGS.expand_softness = float(defaults.get("expand_image", {}).get("softness", 100.))
    DEFAULT_SAMPLE_SETTINGS.expand_space = float(defaults.get("expand_image", {}).get("space", 15.))
    DEFAULT_SAMPLE_SETTINGS.expand_top = float(defaults.get("expand_image", {}).get("top", 0.))
    DEFAULT_SAMPLE_SETTINGS.expand_bottom = float(defaults.get("expand_image", {}).get("bottom", 0.))
    DEFAULT_SAMPLE_SETTINGS.expand_left = float(defaults.get("expand_image", {}).get("left", 0.))
    DEFAULT_SAMPLE_SETTINGS.expand_right = float(defaults.get("expand_image", {}).get("right", 0.))

    DEFAULT_SAMPLE_SETTINGS.start_time = ""
    DEFAULT_SAMPLE_SETTINGS.end_time = ""
    DEFAULT_SAMPLE_SETTINGS.elapsed_time = ""
    DEFAULT_SAMPLE_SETTINGS.uuid = ""              # randomly generated string unique to the sample(s), not used
    DEFAULT_SAMPLE_SETTINGS.status = 0             # waiting to be processed
    DEFAULT_SAMPLE_SETTINGS.error_message = ""     # if there is an error, this will have relevant information
    DEFAULT_SAMPLE_SETTINGS.output_path = ""       # by default an output path based on the prompt will be used
    DEFAULT_SAMPLE_SETTINGS.output_name = ""       # by default an output name based on the prompt will be used
    DEFAULT_SAMPLE_SETTINGS.output_file = ""       # if sampling is successful this is the path to the output image file
    DEFAULT_SAMPLE_SETTINGS.output_sample = None   # cv2 image of the output sample if sampling is successful (or None)
    DEFAULT_SAMPLE_SETTINGS.output_expand_mask  = None # if expand parameters were used this is the mask used to expand the image
    DEFAULT_SAMPLE_SETTINGS.no_output_file = False # if True do not save an output file for the sample
    DEFAULT_SAMPLE_SETTINGS.args_file = ""         # path to a file containing the arguments used for sampling
    DEFAULT_SAMPLE_SETTINGS.no_args_file = False   # if True do not save a separate args file for the output sample
    DEFAULT_SAMPLE_SETTINGS.debug = False

    return

def get_default_args():
    global DEFAULT_SAMPLE_SETTINGS
    default_args = Namespace(**vars(DEFAULT_SAMPLE_SETTINGS))
    return default_args # copy the default settings

def save_yaml(_dict, file_path):
    (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    with open(file_path, "w") as file:
        yaml.dump(_dict, file, sort_keys=False)
    return

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.load(file, Loader=Loader)
    return data

def get_random_string(digits=8):
    uuid_str = str(uuid.uuid4())
    return uuid_str[0:digits] # shorten uuid, don't need that many digits usually

def print_args(args, verbosity_level=1, return_only=False, width=0):
    if type(args) == Namespace:
        namespace_dict = vars(strip_args(args, level=verbosity_level))
    elif type(args) == list:
        namespace_dict = []
        for arg in args:
            namespace_dict.append(vars(strip_args(arg, level=verbosity_level)))
    else:
        raise("args must be an arguments Namespace or list of Namespaces")
    if width > 0: arg_string = yaml.dump(namespace_dict, sort_keys=False, width=width)
    else: arg_string = yaml.dump(namespace_dict, sort_keys=False)
    if not return_only: print(arg_string)
    return arg_string

def strip_args(args, level=1): # remove args we wouldn't want to print or serialize, higher levels strip additional irrelevant fields
    stripped = argparse.Namespace(**(vars(args)))

    # for level 0 only strip fields that can't / shouldn't be serialized
    if "output_sample" in stripped:
        del stripped.output_sample
    if "outout_expand_mask" in stripped:
        del stripped.outout_expand_mask
    if "init_image" in stripped:
        if type(stripped.init_image) != str: del stripped.init_image

    if level >=1: # keep just the basics for most printing

        if "auto_seed_low" in stripped: del stripped.auto_seed_low
        if "auto_seed_high" in stripped: del stripped.auto_seed_high
        if "max_width" in stripped: del stripped.max_width
        if "max_height" in stripped: del stripped.max_height
                
        if "debug" in stripped:
            if not stripped.debug: del stripped.debug
        if "seed" in stripped:
            if stripped.seed == 0: del stripped.seed
        if "auto_seed" in stripped:
            if stripped.auto_seed == 0: del stripped.auto_seed            
        if "elapsed_time" in stripped:
            if stripped.elapsed_time == "": del stripped.elapsed_time
        if "negative_prompt" in stripped:
            if stripped.negative_prompt == "": del stripped.negative_prompt            
        if "output_path" in stripped:
            if stripped.output_path == "": del stripped.output_path
        if "output_name" in stripped:
            if stripped.output_name == "": del stripped.output_name
        if "output_file" in stripped:
            if stripped.output_file == "": del stripped.output_file
        if "error_message" in stripped:
            if stripped.error_message == "": del stripped.error_message
        if "args_file" in stripped:
            if stripped.args_file == "": del stripped.args_file

        if "start_time" in stripped: del stripped.start_time
        if "end_time" in stripped: del stripped.end_time
        if "no_output_file" in stripped: del stripped.no_output_file
        if "no_args_file" in stripped: del stripped.no_args_file
        if "uuid" in stripped: del stripped.uuid
        if "status" in stripped: del stripped.status

        if "init_image" in stripped:
            if stripped.init_image == "": # if there was no input image these fields are not relevant
                del stripped.init_image
        if "init_image" not in stripped:                
            if "img2img_strength" in stripped: del stripped.img2img_strength
            if "expand_softness" in stripped: del stripped.expand_softness
            if "expand_space" in stripped: del stripped.expand_space
            if "expand_top" in stripped: del stripped.expand_top
            if "expand_bottom" in stripped: del stripped.expand_bottom
            if "expand_left" in stripped: del stripped.expand_left
            if "expand_right" in stripped: del stripped.expand_right

    if level >=2: # strip what the discord bot won't need to echo
        if "init_image" in stripped:
            del stripped.init_image
        if "elapsed_time" in stripped:
            del stripped.elapsed_time
        if "output_file" in stripped:
            del stripped.output_file
        if "args_file" in stripped:
            del stripped.args_file
        if "error_message" in stripped:
            del stripped.error_message
        if "auto_seed" in stripped:
            stripped.seed = stripped.auto_seed
            del stripped.auto_seed

        # note - disabling this for now, I think the verbosity might be better
        # remove any values that match value defaults
        """
        for key, value in vars(DEFAULT_SAMPLE_SETTINGS).items():
            if key in stripped:
                if value == stripped.__dict__[key]:
                    delattr(stripped, key)
        """

    return stripped

def validate_resolution(args):      # clip output dimensions at max_resolution, while keeping the correct resolution granularity,
    global DEFAULT_SAMPLE_SETTINGS  # while roughly preserving aspect ratio.
    RESOLUTION_GRANULARITY = 8      # hard-coding this for now, not likely to change any time soon

    width, height = (args.width, args.height)
    aspect_ratio = width / height 
    if width > args.max_width and args.max_width > 0:
        width = args.max_width
        height = int(width / aspect_ratio)
    if height > args.max_height and args.max_height > 0:
        height = args.max_height
        width = int(height * aspect_ratio)
        
    width = int(width / RESOLUTION_GRANULARITY) * RESOLUTION_GRANULARITY
    height = int(height / RESOLUTION_GRANULARITY) * RESOLUTION_GRANULARITY
    width = max(width, RESOLUTION_GRANULARITY)
    height = max(height, RESOLUTION_GRANULARITY)

    args.width, args.height = (width, height)
    return

def soften_mask(np_rgba_image, softness, space):
    if softness == 0: return np_rgba_image
    original_max_opacity = np.max(np_rgba_image[:,:,3])
    out_mask = np_rgba_image[:,:,3] <= 0.
    blurred_mask = gdl_utils.gaussian_blur(np_rgba_image[:,:,3], 3.14/softness, mode="linear_gradient")
    blurred_mask = np.maximum(blurred_mask - np.max(blurred_mask[out_mask]), 0.) 
    np_rgba_image[:,:,3] *= blurred_mask  # preserve partial opacity in original input mask
    np_rgba_image[:,:,3] /= np.max(np_rgba_image[:,:,3]) # renormalize
    np_rgba_image[:,:,3] = np.clip(np_rgba_image[:,:,3] - space, 0., 1.) # make space
    np_rgba_image[:,:,3] /= np.max(np_rgba_image[:,:,3]) # and renormalize again
    np_rgba_image[:,:,3] *= original_max_opacity # restore original max opacity
    return np_rgba_image                 

def expand_image(cv2_img, top, right, bottom, left, softness, space):
    top = int(top / 100. * cv2_img.shape[0])
    right = int(right / 100. * cv2_img.shape[1])
    bottom = int(bottom / 100. * cv2_img.shape[0])
    left = int(left / 100. * cv2_img.shape[1])
    new_width = cv2_img.shape[1] + left + right
    new_height = cv2_img.shape[0] + top + bottom
    new_img = np.zeros((new_height, new_width, 4), np.uint8) # expanded image is rgba

    print("Expanding input image from {0}x{1} to {2}x{3}".format(cv2_img.shape[1], cv2_img.shape[0], new_width, new_height))
    if cv2_img.shape[2] == 3: # rgb input image
        new_img[top:top+cv2_img.shape[0], left:left+cv2_img.shape[1], 0:3] = cv2_img
        new_img[top:top+cv2_img.shape[0], left:left+cv2_img.shape[1], 3] = 255 # fully opaque
    elif cv2_img.shape[2] == 4: # rgba input image
        new_img[top:top+cv2_img.shape[0], left:left+cv2_img.shape[1]] = cv2_img
    else:
        raise Exception("Unsupported image format: {0} channels".format(cv2_img.shape[2]))
        
    if softness > 0.:
        #save_image(new_img, "temp/debug_expanded_pre-soften.png") #debug
        new_img = soften_mask(new_img/255., softness/100., space/100.)
        new_img = (np.clip(new_img, 0., 1.)*255.).astype(np.uint8)
        #save_image(new_img, "temp/debug_expanded.png") #debug

    return new_img

def save_image(cv2_image, file_path):
    (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    cv2.imwrite(file_path, cv2_image)
    return

def load_image(file_path, cv2_flags=cv2.IMREAD_UNCHANGED):
    return cv2.imread(file_path, cv2_flags)

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

def prepare_init_image(args):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS
    MINIMUM_OUTPAINT_IMG2IMG_STRENGTH = 1. # 2.
    #CV2_RESIZE_INTERPOLATION_MODE = cv2.INTER_LANCZOS4
    CV2_RESIZE_INTERPOLATION_MODE = cv2.INTER_AREA

    # load and resize input image to multiple of 8x8
    if type(args.init_image) == str:
        fs_init_image_path = (pathlib.Path(DEFAULT_PATHS.inputs) / args.init_image).as_posix()
        init_image = load_image(fs_init_image_path)
    else:
        init_image = args.init_image

    if args.expand_top or args.expand_bottom or args.expand_left or args.expand_right:
        init_image = expand_image(init_image, args.expand_top, args.expand_right, args.expand_bottom, args.expand_left, args.expand_softness, args.expand_space)
    args.width, args.height = (init_image.shape[1], init_image.shape[0])
    validate_resolution(args)
    num_channels = init_image.shape[2]

    if (args.width, args.height) != (init_image.shape[1], init_image.shape[0]):  # todo: implement mask-aware rescaler
        print("Resizing input image from {0}x{1} to {2}x{3}".format(init_image.shape[1], init_image.shape[0], args.width, args.height))
        # resizing an image with a mask has been a PITA, but cv2.INTER_AREA seems to cooperate if you're downsampling exclusively
        init_image = np.clip(cv2.resize(init_image, (args.width, args.height), interpolation=CV2_RESIZE_INTERPOLATION_MODE), 0, 255)

    if num_channels == 4:  # input image has an alpha channel, setup mask for in/out-painting
        args.img2img_strength = float(np.maximum(MINIMUM_OUTPAINT_IMG2IMG_STRENGTH, args.img2img_strength))
        mask_image = 255. - init_image[:,:,3] # extract mask from alpha channel and invert
        init_image = 0. + init_image[:,:,0:3] # strip mask from init_img leaving only rgb channels
        #init_image *= gdl_utils.np_img_grey_to_rgb(mask_image) < 255. # force color data in erased areas to 0
        #mask_image = gdl_utils.np_img_grey_to_rgb(mask_image)

        print("Using in/out-painting with strength {0}".format(args.img2img_strength))
        if args.sampler == "k_euler": # k_euler currently does not add noise during sampling
            print("Warning: k_euler is not currently supported for in-painting, switching to sampler=k_euler_ancestral")
            args.sampler = "k_euler_ancestral"
        
    elif num_channels == 3: # rgb image, regular img2img without a mask
        print("Using img2img with strength {0}".format(args.img2img_strength))
        mask_image = None
    else:
        raise Exception("Error loading init_image "+fs_init_image_path+": unsupported image format")

    return init_image, mask_image

async def get_samples(args, interactive=False):
    global DEFAULT_PATHS, GRPC_SERVER_SETTINGS

    if (not interactive) and (args.num_samples <= 0):
        raise Exception("Error: num_samples must be > 0 for non-interactive sampling")
    if args.num_samples <= 0: print("Repeating sample, press ctrl+c to stop...")

    printed_waiting_message = False
    while True: # ensure selected model is available and loaded before proceeding
        models = get_models()
        if models == None:
            raise Exception("Error: SDGRPC server is unavailable")
        model_found = False
        model_ready = False
        for model in models:
            if model["id"] == args.model_name:
                model_found = True
                if model["ready"] == True: model_ready = True
                break
        if not model_found:
            show_models(models)
            raise Exception("Error: model '"+args.model_name+"' not found on server")
        if not model_ready:
            if not printed_waiting_message: # once is probably enough
                print("Waiting for model '{0}' to be ready...".format(args.model_name))
                printed_waiting_message = True
            await asyncio.sleep(1.5)
            continue
        break

    def get_samples_wrapper(args):
        thread = threading.current_thread()
        thread.args = args # always return at least the original args if _get_samples doesn't return
        thread.args = _get_samples(args)
        return
    sample_thread = Thread(target = get_samples_wrapper, args=[args], daemon=True)

    def cancel_request(unused_signum, unused_frame, sample_thread):
        print("Okay, cancelling...")
        if hasattr(sample_thread, "args"):
            if type(sample_thread.args) == Namespace: args = sample_thread.args
            elif type(sample_thread.args) == list: args = sample_thread.args[-1]
            args.status = 3 # cancelled
            args.error_message = "Cancelled by user"

        if sample_thread.ident:
            import ctypes        # this is a a bit of a hack to instantly stop the sample thread but it works
            _stderr = sys.stderr # required to suppress the exception printing from the sample_thread
            sys.stderr = open('nul', 'w')
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(sample_thread.ident), ctypes.py_object(KeyboardInterrupt))
            sample_thread.join() # ensure the thread is dead
            sys.stderr = _stderr # restore stderr
        else:
            raise KeyboardInterrupt
        return
    import signal, functools # this ugly hack is required because of asyncio and keyboardinterrupt behavior
    signal.signal(signal.SIGINT, functools.partial(cancel_request, sample_thread=sample_thread))

    try:
        global GRPC_SERVER_LOCK
        async with GRPC_SERVER_LOCK: # only one grpc request at a time please (at least, in this process)
            sample_thread.start()
            while True:
                sample_thread.join(0.)
                if not sample_thread.is_alive(): break
                await asyncio.sleep(0.1)
            sample_thread.join() # it's the only way to be sure
    except Exception as e:
        raise
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL) # ensure the original handler is properly restored

    return sample_thread.args

def build_grpc_request_dict(args, init_image_bytes, mask_image_bytes):
    # use auto-seed if none provided
    if args.seed: seed = args.seed
    else: seed = args.auto_seed

    # sample indefinitely if num_samples is 0
    if args.num_samples > 0: num_samples = args.num_samples
    else: num_samples = int(1e6) # effectively infinite

    prompt = args.prompt if args.prompt != "" else " "
    return {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "start_schedule": args.img2img_strength,
        "end_schedule": 0.01,
        "cfg_scale": args.cfg_scale,
        "eta": 0.0,
        "sampler": grpc_client.get_sampler_from_str(args.sampler),
        "steps": args.steps,
        "seed": seed,
        "samples": num_samples,
        "init_image": init_image_bytes,
        "mask_image": mask_image_bytes,
        "negative_prompt": args.negative_prompt,
        "guidance_preset": grpc_client.generation.GUIDANCE_PRESET_SIMPLE if args.guidance_strength > 0. else grpc_client.generation.GUIDANCE_PRESET_NONE,
        "guidance_strength": args.guidance_strength,
        "guidance_prompt": prompt,
        "hires_fix": args.hires_fix,
        "tiling": args.seamless_tiling,
    }

def _get_samples(args):
    global DEFAULT_SAMPLE_SETTINGS
    
    # set auto-seed if no seed provided
    if args.seed == 0:
        if args.auto_seed == 0: # no existing auto-seed, use new seed from auto-seed range
            args.auto_seed = int(np.random.randint(args.auto_seed_low, args.auto_seed_high)) # new random auto-seed
    else:
        args.auto_seed = 0 # seed provided, disable auto-seed

    # load input image if we have one
    if type(args.init_image) == str:
        load_init_image = args.init_image != ""
    else:
        load_init_image = True

    if load_init_image:
        init_image, mask_image = prepare_init_image(args)
    else:
        init_image, mask_image = (None, None)
        validate_resolution(args)
    if init_image is None: init_image_bytes = None # pre-encode images for grpc request
    else: init_image_bytes = np.array(cv2.imencode(".png", init_image)[1]).tobytes()
    if mask_image is None: mask_image_bytes = None
    else: mask_image_bytes = np.array(cv2.imencode(".png", mask_image)[1]).tobytes()

    output_args = []
    stability_api = grpc_client.StabilityInference(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.grpc_key, engine=args.model_name, verbose=False, wait_for_ready=False)
    
    def _reset_args(args):
        args.status = 1 # in progress
        args.error_message = ""
        args.output_file = ""
        args.output_sample = None
        args.output_expand_mask = None
        args.args_file = ""

        start_time = datetime.datetime.now()
        args.start_time = str(start_time)
        args.end_time = ""
        args.elapsed_time = ""
        args.uuid = get_random_string(digits=16) # new uuid for new sample
        return start_time

    try:
        request_dict = build_grpc_request_dict(args, init_image_bytes, mask_image_bytes)
        answers = stability_api.generate(**request_dict)
        grpc_samples = grpc_client.process_artifacts_from_answers("", answers, write=False, verbose=False)

        start_time = _reset_args(args)
        for path, artifact in grpc_samples:
            image = cv2.imdecode(np.fromstring(artifact.binary, dtype="uint8"), cv2.IMREAD_UNCHANGED)

            if not (mask_image is None): # blend original image back in for in/out-painting, this is required due to vae decoding artifacts
                mask_rgb = 1.-gdl_utils.np_img_grey_to_rgb(mask_image/255.)
                image = np.clip(image*(1.-mask_rgb) + init_image*mask_rgb, 0., 255.)
                args.output_expand_mask = 255-mask_image
            if "annotation" in args: image = get_annotated_image(image, args)

            end_time = datetime.datetime.now()
            args.end_time = str(end_time)
            args.elapsed_time = str(end_time-start_time)
            args.status = 2 # completed successfully
            args.output_sample = image

            save_sample(image, args)
            output_args.append(Namespace(**vars(args)))

            if args.seed: args.seed += 1 # increment seed or random seed if none was given as we go through the batch
            else: args.auto_seed += 1
            if (len(output_args) < args.num_samples) or (args.num_samples <= 0): 
                start_time = _reset_args(args) # reset start time and status for next sample if we still have samples left

    except KeyboardInterrupt:
        args.status = 3
        args.error_message = "Cancelled by user"
        try: answers.cancel()
        except: pass
    except Exception as e:
        args.status = -1 # error status
        args.error_message = str(e)
        print("Error in grpc sample request: ", e)

    return output_args

def get_default_output_name(name, truncate_length=100, force_ascii=True):
    if force_ascii: name = str(name.encode('utf-8').decode('ascii', 'ignore'))
    sanitized_name = re.sub(r'[\\/*?:"<>|]',"", name).replace(".","").replace("'","").replace("\t"," ").replace(" ","_").strip()
    if (truncate_length > len(sanitized_name)) or (truncate_length==0): truncate_length = len(sanitized_name)
    if truncate_length < len(sanitized_name):  sanitized_name = sanitized_name[0:truncate_length]
    return sanitized_name

def get_noclobber_checked_path(base_path, file_path):
    clobber_num_padding = 2
    file_path_noext, file_path_ext = os.path.splitext(file_path)
    existing_count = len(glob.glob(base_path+"/"+file_path_noext+"*"+file_path_ext))
    return file_path_noext+"_x"+str(existing_count).zfill(clobber_num_padding)+file_path_ext

def get_fs_paths(args): # get filesystem output paths / base filenames from args
    if not args.prompt: prompt = " "
    else: prompt = args.prompt
    if not args.output_name: fs_output_name = get_default_output_name(prompt)
    else: fs_output_name = get_default_output_name(args.output_name)
    if not args.output_path: fs_output_path = fs_output_name
    else: fs_output_path = get_default_output_name(args.output_path)
    return fs_output_name, fs_output_path

def save_sample(sample, args):
    global DEFAULT_PATHS

    fs_output_name, fs_output_path = get_fs_paths(args)
    if args.seed: seed = args.seed
    else: seed = args.auto_seed
    seed_num_padding = 5

    filename = fs_output_name+"_s"+str(seed).zfill(seed_num_padding)+".png"
    args.output_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, fs_output_path+"/"+filename)
    full_output_path = DEFAULT_PATHS.outputs+"/"+args.output_file
    save_image(sample, full_output_path)
    print("Saved {0}".format(full_output_path))

    if not args.no_args_file:
        args.args_file = fs_output_path+"/args/"+fs_output_name+"_s"+str(seed).zfill(seed_num_padding)+".yaml"
        args.args_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, args.args_file) # add suffix if filename already exists
        full_args_path = DEFAULT_PATHS.outputs+"/"+args.args_file
        save_yaml(vars(strip_args(args, level=0)), full_args_path)
        #print("Saved {0}".format(full_args_path)) # make this silent for now, it's a bit spammy
    return args.output_file

def save_image_grid(images, file_path):
    global DEFAULT_PATHS
    grid_layout = get_grid_layout(len(images))
    grid_image = get_image_grid(images, grid_layout)
    output_file = DEFAULT_PATHS.outputs+"/"+get_noclobber_checked_path(DEFAULT_PATHS.outputs, file_path)
    save_image(grid_image, output_file)
    print("Saved grid image {0}".format(output_file))
    return
