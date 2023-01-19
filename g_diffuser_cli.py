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


g_diffuser_cli.py - interactive command line interface for g-diffuser

"""

import modules.g_diffuser_lib as gdl
from modules.g_diffuser_lib import SimpleLogger

import argparse
from argparse import Namespace
import code
import glob
import pathlib
import asyncio
import os

import numpy as np
import cv2

VERSION_STRING = "g-diffuser-cli v2.0"
INTERACTIVE_MODE_BANNER_STRING = """Interactive mode:
    call sample() with keyword arguments and use the up/down arrow-keys to browse command history:

sample("pillars of creation", num_samples=3, cfg_scale=15)          # batch of 3 samples with cfg scale 15
sample("greg rutkowski", init_image="my_image.png", num_samples=0)  # setting n <=0 repeats until stopped
sample("something's wrong with the g-diffuser", sampler="k_euler")  # uses the k_euler sampler
                                                                    # any parameters unspecified will use defaults

show_args(default_args())            # show default arguments and sampling parameters
my_args = default_args(cfg_scale=15) # you can assign a collection of arguments to a variable
my_args.prompt = "art by frank"      # and modify them before passing them to sample()
sample(my_args)                      # sample modifies the arguments object you pass in with results
show_args(my_args)                   # you can show the result args to verify the results and output path

show_samplers() # show all available sampler names
show_models()   # show available model ids on the grpc server (check models.yaml for more info)
save_args(my_args, "my_fav_args")  # you can save your arguments in /inputs/args
args = load_args("my_fav_args")    # you can load those saved arguments by name

run_script("zoom_maker", my_zoom_args)    # you can save cli scripts(.py) in /inputs/scripts and run them in the cli
run("zoom_composite", my_composite_args)  # run is shorthand for run_script, you can also pass an args object to the script

resample("old_path", "new_path", scale=20)  # regenerate all saved outputs in /outputs/old_path into /outputs/new_path
                                            # with optional replacement / substituted arguments
compare("path1", "path2", "path3")          # make a comparison grid from all images in the specified output paths
compare("a", "b", mode="rows")              # arrange each output path's images into rows instead
compare("a", "b", file="my_compare.jpg")    # the comparison image will be saved by default as /outputs/compare.jpg
                                            # use 'file' to specify an alternate filename

clear()        # clear the command window history
help()         # display this message
exit()         # exit interactive mode
"""

def main():
    gdl.load_config()
    os.chdir(gdl.DEFAULT_PATHS.root)
    logger = SimpleLogger("g_diffuser_cli.log")
    
    gdl.start_grpc_server()
    
    global LAST_ARGS_PATH
    LAST_ARGS_PATH = gdl.DEFAULT_PATHS.inputs+"/args/last_args.yaml"

    global cli_locals
    cli_locals = argparse.Namespace()
    cli_locals.sample = cli_get_samples
    cli_locals.s = cli_get_samples
    cli_locals.show_args = cli_show_args
    cli_locals.sha = cli_show_args
    cli_locals.load_args = cli_load_args
    cli_locals.la = cli_load_args
    cli_locals.save_args = cli_save_args
    cli_locals.sa = cli_save_args
    cli_locals.default_args = cli_default_args
    cli_locals.resample = cli_resample
    cli_locals.compare = cli_save_comparison_grid
    cli_locals.run_script = cli_run_script
    cli_locals.run = cli_run_script
    cli_locals.show_samplers = cli_show_samplers
    cli_locals.show_models = cli_show_models
    cli_locals.clear = cli_clear
    cli_locals.help = cli_help
    cli_locals.exit = cli_exit
    
    global INTERACTIVE_MODE_BANNER_STRING
    interpreter = code.InteractiveConsole(locals=dict(globals(), **vars(cli_locals)))
    interpreter.interact(banner=INTERACTIVE_MODE_BANNER_STRING, exitmsg="")

    return
    
def cli_get_samples(prompt=None, **kwargs):
    global LAST_ARGS_PATH

    if prompt is None: prompt = ""
    if type(prompt) == argparse.Namespace: # prompt can be a prompt string or an args namespace
        args = prompt
    elif type(prompt) == str:
        args = cli_default_args(prompt=prompt)
    else:
        raise Exception("Invalid prompt type: {0} - '{1}'".format(str(type(prompt)), str(prompt)))

    args.__dict__ |= kwargs # merge / override with kwargs
    asyncio.run(gdl.get_samples(args, interactive=True))

    # try to save the last used args in a file for convenience
    try: gdl.save_yaml(vars(gdl.strip_args(args, level=0)), LAST_ARGS_PATH)
    except Exception as e: pass
    return
    
def cli_show_args(args, level=None):
    if level != None: verbosity_level = level
    else: verbosity_level = 1
    gdl.print_args(args, verbosity_level=verbosity_level)
    return
    
def cli_load_args(name=""):
    global LAST_ARGS_PATH
    try:
        if not name: args_path = LAST_ARGS_PATH
        else: args_path = gdl.DEFAULT_PATHS.inputs+"/args/"+name+".yaml"
        saved_args = Namespace(**gdl.load_yaml(args_path))
        gdl.print_args(saved_args)
    except Exception as e:
        print("Error loading args from file - {0}".format(e))
    return saved_args
    
def cli_save_args(args, name):
    try:
        args_path = gdl.DEFAULT_PATHS.inputs+"/args/"+name+".yaml"
        gdl.save_yaml(vars(gdl.strip_args(args, level=0)), args_path)
        print("Saved {0}".format(args_path))
    except Exception as e:
        print("Error saving args - {0}".format(e))
    return

def cli_default_args(**kwargs):
    return Namespace(**(vars(gdl.get_default_args()) | kwargs))

def cli_resample(old_path, new_path, **kwargs):
    resample_args = argparse.Namespace(**kwargs)
    assert(old_path); assert(new_path)
    global DEFAULT_PATHS
    
    if not os.path.exists(DEFAULT_PATHS.outputs+"/"+old_path):
        print("Error: Output path '" + str(DEFAULT_PATHS.outputs+"/"+old_path) + "' does not exist")
        return

    all_resampled_samples = []
    old_arg_files = glob.glob(DEFAULT_PATHS.outputs+"/"+old_path+"/**/*.json", recursive=True)
    if len(old_arg_files) > 0:
        print("Resampling "+str(len(old_arg_files)) + " output samples...")
        for arg_file in old_arg_files:
            args_file_dict = gdl.load_json(arg_file); assert(args_file_dict)
            if args_file_dict["n"] < 1: continue # skip samples that were endlessly repeated

            output_resample_args = argparse.Namespace(**(args_file_dict | vars(resample_args))) # merge with original args
            output_resample_args.n = 1
            output_resample_args.output_path = new_path # ensure output goes to specified path, regardless of output_path in args

            try:
                samples = gdl.get_samples(output_resample_args)
                all_resampled_samples.extend(samples)
            except KeyboardInterrupt: 
                print("Aborting resample...")
                return
            except Exception as e:
                print("Error in gdl.get_samples '" + str(e) + "'")
    else:
        print("No outputs found in '" + str(DEFAULT_PATHS.outputs+"/"+old_path) + "' to resample")

    return
    
def cli_save_comparison_grid(*paths, **kwargs):
    global DEFAULT_PATHS
    args = argparse.Namespace(**kwargs)
    if not "mode" in args: args.mode="columns"
    else: args.mode = args.mode.lower()
    if not "file" in args: grid_filename = "compare.jpg"
    else: grid_filename = args.file
    if "compare_output_path" in args:
        if args.compare_output_path:
            grid_filename = DEFAULT_PATHS.outputs+"/"+args.compare_output_path+"/"+grid_filename
        else:
            grid_filename = DEFAULT_PATHS.outputs+"/"+grid_filename
    else:
        grid_filename = DEFAULT_PATHS.outputs+"/"+grid_filename

    num_paths = len(paths)
    path_samples = []

    max_sample_width = 0  # keep track of the largest image in all the folders to make everything fit in the event of non-uniform size
    max_sample_height = 0
    for path in paths:
        assert(type(path) == str)
        path_files = glob.glob(DEFAULT_PATHS.outputs+"/"+path+"/*.png")
        for file in path_files:
            if os.path.basename(file).startswith("grid_"): path_files.remove(file) # exclude grid images from comparison grids
        path_files = sorted(path_files)

        samples = []
        for file in path_files:
            img = cv2.imread(file)
            max_sample_width = np.maximum(max_sample_width, img.shape[0])
            max_sample_height = np.maximum(max_sample_height, img.shape[1])
            samples.append(img)
        path_samples.append(samples)

    max_path_samples = 0
    for path_sample_list in path_samples:
        if len(path_sample_list) > max_path_samples: max_path_samples = len(path_sample_list)

    if args.mode != "rows": layout = (max_path_samples, num_paths)
    else: layout = (num_paths, max_path_samples)
    np_grid = np.zeros((layout[0] * max_sample_width, layout[1] * max_sample_height, 3), dtype="uint8")

    for x in range(len(path_samples)):
        for y in range(len(path_samples[x])):
            sample = path_samples[x][y]
            paste_x = x * max_sample_width
            paste_y = y * max_sample_height
            if args.mode != "rows":
                paste_x = y * max_sample_width
                paste_y = x * max_sample_height
            np_grid[paste_x:paste_x+max_sample_width, paste_y:paste_y+max_sample_height, :] = sample[:]
    
    (pathlib.Path(grid_filename).parents[0]).mkdir(exist_ok=True, parents=True)
    cv2.imwrite(grid_filename, np_grid)
    print("Saved " + grid_filename)
    return

def cli_run_script(script_name, args=None, **kwargs):
    global cli_locals
    script_path = gdl.DEFAULT_PATHS.inputs+"/scripts/"+script_name+".py"
    args_dict = {"cli_args": args, "kwargs": kwargs}
    script_locals = dict(globals(), **(vars(cli_locals) | args_dict))
    try:
        with open(script_path, "r") as script_file:
            exec(script_file.read(), script_locals)
    except KeyboardInterrupt:
        print("Okay, cancelling...")
    except Exception as e:
        raise

    if args: # if an args object was passed, update the object with the results from the script
        result_args = script_locals.get("args", None)
        if type(result_args) == Namespace:
            args.__dict__ = result_args.__dict__    
    return

def cli_show_samplers():
    for sampler in gdl.GRPC_SERVER_SUPPORTED_SAMPLERS_LIST:
        print("sampler='"+sampler+"'")
    return

def cli_show_models():
    gdl.show_models()
    return

def cli_clear():
    if os.name == "nt": os.system("cls")
    else: os.system("clear")
    return
    
def cli_help():
    global VERSION_STRING, INTERACTIVE_MODE_BANNER_STRING
    print(VERSION_STRING+INTERACTIVE_MODE_BANNER_STRING+"\n")
    return
    
def cli_exit():
    exit(0)
    
    
if __name__ == "__main__":
    main()