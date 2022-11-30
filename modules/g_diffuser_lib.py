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

import yaml
from yaml import CLoader as Loader

import numpy as np
import cv2

from modules import sdgrpcserver_client as grpc_client
from modules import g_diffuser_utilities as gdl_utils

GRPC_SERVER_SUPPORTED_SAMPLERS_LIST = list(grpc_client.SAMPLERS.keys())
GRPC_SERVER_ENGINE_STATUS = []
GRPC_SERVER_LOCK = asyncio.Lock()

def start_grpc_server():
    global GRPC_SERVER_SETTINGS
    if get_socket_listening_status(GRPC_SERVER_SETTINGS.host):
        print("\nFound running SDGRPC server listening on {0}".format(GRPC_SERVER_SETTINGS.host))
    else:
        raise Exception("Could not connect to SDGRPC server at {0}, is the server running?".format(GRPC_SERVER_SETTINGS.host))

    try:
        models = get_models()
        print("Found {0} model(s) on the SDGRPC server:".format(len(models)))
        print(yaml.dump(models, sort_keys=False))
        all_models_ready = True
        for model in models:
            all_models_ready &= model["ready"]
        if not all_models_ready:
            print("Not all models are ready yet, the server may still be starting up...")

    except Exception as e:
        raise Exception("Unable to query server status at {0} with key'{1}', is the host/key correct?".format(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.grpc_key))

    return

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

def load_config():
    global DEFAULT_PATHS
    global GRPC_SERVER_SETTINGS
    global DISCORD_BOT_SETTINGS
    global DEFAULT_SAMPLE_SETTINGS

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
    GRPC_SERVER_SETTINGS.nsfw_behavior = os.environ.get("SD_NSFW_BEHAVIOUR", "block")
    GRPC_SERVER_SETTINGS.vram_optimization_level = int(os.environ.get("SD_VRAM_OPTIMISATION_LEVEL", "2"))
    if GRPC_SERVER_SETTINGS.host == "localhost":
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
    except:
        print("Warning: Could not load defaults file {0}".format(DEFAULT_PATHS.defaults))
        defaults = {}

    DEFAULT_SAMPLE_SETTINGS = argparse.Namespace()
    DEFAULT_SAMPLE_SETTINGS.prompt = ""
    DEFAULT_SAMPLE_SETTINGS.seed = 0 # 0 means use auto seed    
    DEFAULT_SAMPLE_SETTINGS.model_name = str(defaults.get("model_name", "default"))
    DEFAULT_SAMPLE_SETTINGS.num_samples = int(defaults.get("num_samples", 1))
    DEFAULT_SAMPLE_SETTINGS.sampler = str(defaults.get("sampler", "dpmspp_2"))
    DEFAULT_SAMPLE_SETTINGS.steps = int(defaults.get("steps", 50))
    DEFAULT_SAMPLE_SETTINGS.max_steps = int(defaults.get("max_steps", 150))    
    DEFAULT_SAMPLE_SETTINGS.cfg_scale = float(defaults.get("cfg_scale", 14.))
    DEFAULT_SAMPLE_SETTINGS.guidance_strength = float(defaults.get("guidance_strength", 0.5))
    DEFAULT_SAMPLE_SETTINGS.negative_prompt = str(defaults.get("negative_prompt", ""))

    DEFAULT_SAMPLE_SETTINGS.width = int(defaults.get("default_resolution", {}).get("width", 512))
    DEFAULT_SAMPLE_SETTINGS.height = int(defaults.get("default_resolution", {}).get("height", 512))
    DEFAULT_SAMPLE_SETTINGS.max_width = int(defaults.get("max_resolution", {}).get("width", 960))
    DEFAULT_SAMPLE_SETTINGS.max_height = int(defaults.get("max_resolution", {}).get("height", 960))
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
    DEFAULT_SAMPLE_SETTINGS.uuid = ""             # randomly generated string unique to the sample(s)
    DEFAULT_SAMPLE_SETTINGS.status = 0            # waiting to be processed
    DEFAULT_SAMPLE_SETTINGS.error_message = ""    # if there is an error, this will have relevant information
    DEFAULT_SAMPLE_SETTINGS.output_path = ""      # by default an output path based on the prompt will be used
    DEFAULT_SAMPLE_SETTINGS.output_name = ""      # by default an output name based on the prompt will be used
    DEFAULT_SAMPLE_SETTINGS.output_file = ""      # if sampling is successful this is the path to the output image file
    DEFAULT_SAMPLE_SETTINGS.args_file = ""        # path to a file containing the arguments used for sampling
    DEFAULT_SAMPLE_SETTINGS.no_args_file = False  # if True do not save a separate args file for the output sample
    DEFAULT_SAMPLE_SETTINGS.debug = False

    return

def get_models():
    global GRPC_SERVER_SETTINGS, GRPC_SERVER_ENGINE_STATUS
    stability_api = grpc_client.StabilityInference(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.grpc_key)
    GRPC_SERVER_ENGINE_STATUS = stability_api.list_engines()
    return GRPC_SERVER_ENGINE_STATUS

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

def print_args(args, verbosity_level=0):
    namespace_dict = vars(strip_args(args, level=verbosity_level))
    print(yaml.dump(namespace_dict, sort_keys=False))
    return

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

def strip_args(args, level=1): # remove args we wouldn't want to print or serialize, higher levels strip additional irrelevant fields
    stripped = argparse.Namespace(**(vars(args)))

    if level >=1: # keep just the basics for most printing
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
        if "no_args_file" in stripped: del stripped.no_args_file
        if "uuid" in stripped: del stripped.uuid
        if "status" in stripped: del stripped.status
        if "auto_seed_low" in stripped: del stripped.auto_seed_low
        if "auto_seed_high" in stripped: del stripped.auto_seed_high
        if "max_width" in stripped: del stripped.max_width
        if "max_height" in stripped: del stripped.max_height

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

    return stripped
    
def get_default_args():
    global DEFAULT_SAMPLE_SETTINGS
    return Namespace(**vars(DEFAULT_SAMPLE_SETTINGS)) # copy the default settings

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

def soften_mask(np_rgba_image, softness, space):
    if softness == 0: return np_rgba_image
    out_mask = np_rgba_image[:,:,3] <= 0.
    blurred_mask = gdl_utils.gaussian_blur(np_rgba_image[:,:,3], 3.14/softness)
    blurred_mask = np.maximum(blurred_mask - np.max(blurred_mask[out_mask]), 0.)
    blurred_mask /= np.max(blurred_mask)
    blurred_mask **= np.maximum(space, 1.)
    np_rgba_image[:,:,3] = blurred_mask
    return np_rgba_image

def expand_image(cv2_img, top, right, bottom, left, softness, space):
    top = int(top / 100. * cv2_img.shape[0])
    right = int(right / 100. * cv2_img.shape[1])
    bottom = int(bottom / 100. * cv2_img.shape[0])
    left = int(left / 100. * cv2_img.shape[1])
    new_width = cv2_img.shape[1] + left + right
    new_height = cv2_img.shape[0] + top + bottom
    new_img = np.zeros((new_height, new_width, 4), np.uint8) # expanded image is rgba

    if cv2_img.shape[2] == 3: # rgb input image
        new_img[top:top+cv2_img.shape[0], left:left+cv2_img.shape[1], 0:3] = cv2_img
        new_img[top:top+cv2_img.shape[0], left:left+cv2_img.shape[1], 3] = 255 # fully opaque
    elif cv2_img.shape[2] == 4: # rgba input image
        new_img[top:top+cv2_img.shape[0], left:left+cv2_img.shape[1]] = cv2_img
    else:
        raise Exception("Unsupported image format: " + str(cv2_img.shape[2]) + " channels")
        
    if softness > 0.:
        new_img = soften_mask(new_img/255., softness/100., space)
        new_img = (np.clip(new_img, 0., 1.)*255.).astype(np.uint8)

    return new_img

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

def save_image(cv2_image, file_path):
    (pathlib.Path(file_path).parents[0]).mkdir(exist_ok=True, parents=True)
    cv2.imwrite(file_path, cv2_image)
    return

def load_image(args):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS
    MASK_CUTOFF_THRESHOLD = 250. #225. #240.     # this will force the image mask to 0 if opacity falls below a threshold. set to 255. to disable

    final_init_img_path = (pathlib.Path(DEFAULT_PATHS.inputs) / args.init_image).as_posix()
    
    # load and resize input image to multiple of 8x8
    init_image = cv2.imread(final_init_img_path, cv2.IMREAD_UNCHANGED) #.astype(np.float64)
    init_image_dims = (init_image.shape[1], init_image.shape[0])
    width, height = validate_resolution(args.w, args.h, init_image_dims)
    if (width, height) != (init_image.shape[1], init_image.shape[0]):
        if args.debug: print("Resizing input image to (" + str(width) + ", " + str(height) + ")")
        #init_image = cv2.resize(init_image, (width, height), interpolation=cv2.INTER_CUBIC)
        init_image = np.clip(cv2.resize(init_image, (width, height), interpolation=cv2.INTER_LANCZOS4), 0, 255)
    args.w = width
    args.h = height
    
    num_channels = init_image.shape[2]
    if num_channels == 4:     # input image has an alpha channel, setup mask for in/out-painting
        args.noise_start = np.maximum(DEFAULT_SAMPLE_SETTINGS.min_outpaint_noise, args.noise_start) # override img2img "strength" if it is < 1., for in/out-painting this should at least 1.
        mask_image = 255. - init_image[:,:,3] # extract mask from alpha channel and invert
        init_image = 0. + init_image[:,:,0:3]      # strip mask from init_img / convert to rgb

        # apply mask cutoff threshold, this is required because bad paint programs corrupt color data by premultiplying alpha
        # using the 8-bit opacity mask, resulting in unexpected artifacts in areas that are almost but not completely erased
        if MASK_CUTOFF_THRESHOLD < 255.:
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

async def get_samples(args):
    global DEFAULT_PATHS, GRPC_SERVER_SETTINGS

    global GRPC_SERVER_LOCK
    async with GRPC_SERVER_LOCK:

        def get_samples_wrapper(args):
            thread = threading.current_thread()
            thread.args = Namespace(**vars(args)) # preserve old args in case of an exception
            thread.args = _get_samples(args)
            return

        sample_thread = Thread(target = get_samples_wrapper, args=[args], daemon=True)
        sample_thread.start()
        try:
            while True:
                sample_thread.join(0.0001)
                if not sample_thread.is_alive(): break
                await asyncio.sleep(0.05)
            sample_thread.join() # it's the only way to be sure
        except KeyboardInterrupt:
            print("Okay, cancelling...")
            sample_thread.raise_exception(KeyboardInterrupt)
            sample_thread.join()
            sample_thread.args.status = -1 # cancelled
            sample_thread.error_message = "Cancelled by user"

    return sample_thread.args

def build_grpc_request_dict(args, init_image_bytes, mask_image_bytes):
    # use auto-seed if none provided
    if args.seed: seed = args.seed
    else: seed = args.auto_seed

    return {
        "prompt": args.prompt if args.prompt != "" else " ",
        "height": args.height,
        "width": args.width,
        "start_schedule": args.img2img_strength,
        "end_schedule": 0.01,
        "cfg_scale": args.cfg_scale,
        "eta": 0.0,
        "sampler": grpc_client.get_sampler_from_str(args.sampler),
        "steps": args.steps,
        "seed": seed,
        "samples": 1,
        "init_image": init_image_bytes,
        "mask_image": mask_image_bytes,
        "negative_prompt": args.negative_prompt,
        "guidance_preset": grpc_client.generation.GUIDANCE_PRESET_SIMPLE if args.guidance_strength > 0. else grpc_client.generation.GUIDANCE_PRESET_NONE,
        "guidance_strength": args.guidance_strength,
        "guidance_prompt": args.prompt,   
    }

def _get_samples(args):
    global DEFAULT_SAMPLE_SETTINGS
    
    # set auto-seed if no seed provided
    if args.seed == 0:
        if args.auto_seed == 0: # no existing auto-seed, use new seed from auto-seed range
            args.auto_seed = int(np.random.randint(DEFAULT_SAMPLE_SETTINGS.auto_seed_low, DEFAULT_SAMPLE_SETTINGS.auto_seed_high)) # new random auto-seed
    else:
        args.auto_seed = 0 # seed provided, disable auto-seed

    if args.init_image != "": # load input image if we have one
        init_image, mask_image = load_image(args)
    else:
        init_image, mask_image = (None, None)
    if init_image is None: init_image_bytes = None # encode images for grpc request
    else: init_image_bytes = np.array(cv2.imencode(".png", init_image)[1]).tobytes()
    if mask_image is None: mask_image_bytes = None
    else: mask_image_bytes = np.array(cv2.imencode(".png", mask_image)[1]).tobytes()

    output_args = []
    stability_api = grpc_client.StabilityInference(GRPC_SERVER_SETTINGS.host, GRPC_SERVER_SETTINGS.grpc_key, engine=args.model_name, verbose=False)
    while True: # watch out! a wild shrew!
        try:
            args.status = 1 # in progress
            args.error_message = ""
            args.output_file = ""
            args.args_file = ""

            start_time = datetime.datetime.now()
            args.start_time = str(start_time)
            args.uuid_str = get_random_string(digits=16) # new uuid for new sample

            request_dict = build_grpc_request_dict(args, init_image, mask_image)
            answers = stability_api.generate(**request_dict)
            grpc_samples = grpc_client.process_artifacts_from_answers("", answers, write=False, verbose=False)

            for path, artifact in grpc_samples:
                image = cv2.imdecode(np.fromstring(artifact.binary, dtype="uint8"), cv2.IMREAD_UNCHANGED)

                if not (mask_image is None): # blend original image back in for in/out-painting, this is required due to vae decoding artifacts
                    mask_rgb = 1.-gdl_utils.np_img_grey_to_rgb(mask_image/255.)
                    image = np.clip(image*(1.-mask_rgb) + init_image*mask_rgb, 0., 255.)

                if "annotation" in args: image = get_annotated_image(image, args)

                end_time = datetime.datetime.now()
                args.end_time = str(end_time)
                args.elapsed_time = str(end_time-start_time)
                args.status = 2 # completed successfully

                save_sample(image, args)

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

    return output_args

def save_sample(sample, args):
    global DEFAULT_PATHS

    # get filesystem output paths / base filenames
    if not args.prompt: prompt = " "
    else: prompt = args.prompt
    if not args.output_name: fs_output_name = get_default_output_name(prompt)
    else: fs_output_name = get_default_output_name(args.output_name)
    if not args.output_path: fs_output_path = fs_output_name
    else: fs_output_path = get_default_output_name(args.output_path)

    if args.seed: seed = args.seed
    else: seed = args.auto_seed
    seed_num_padding = 5
    
    filename = fs_output_name+"_s"+str(seed).zfill(seed_num_padding)+".png"
    args.output_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, fs_output_path+"/"+filename)
    full_output_path = DEFAULT_PATHS.outputs+"/"+args.output_file
    save_image(sample, full_output_path)
    print("Saved {0}".format(full_output_path))

    if not args.no_args_file:
        args.args_file = fs_output_path+"/args/"+args.final_output_name+"_s"+str(seed).zfill(seed_num_padding)+".yaml"
        args.args_file = get_noclobber_checked_path(DEFAULT_PATHS.outputs, args.args_file) # add suffix if filename already exists
        full_args_path = DEFAULT_PATHS.outputs+"/"+args.args_file
        save_yaml(vars(strip_args(args)), full_args_path)
        print("Saved {0}".format(full_args_path))
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