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


g_diffuser_cli.py - command line interface for g-diffuser-lib with interactive mode

"""

import ntpath # these lines are inexplicably required for python to use long file paths on Windows -_-
ntpath.realpath = ntpath.abspath

import g_diffuser_lib as gdl
from g_diffuser_config import DEFAULT_PATHS

import os, sys
os.chdir(DEFAULT_PATHS.root)

import datetime
import argparse
import code
import importlib

VERSION_STRING = "g-diffuser-cli v0.85b"
INTERACTIVE_MODE_BANNER_STRING = """
Interactive mode:
    call sample() with keyword arguments and use the up/down arrow-keys to browse command history:

sample("my prompt", n=3, scale=15) # generate 3 samples with a scale of 15
sample("greg rutkowski", init_img="my_image.png", repeat=1, debug=1) # repeats until stopped, enables debug mode
sample()     # arguments can be omitted to use your last args instead

reset_args() # reset your arguments back to default values
show_args()  # shows your *basic* input arguments
show_args(0) # shows *all* your input arguments
load_args()  # use your last arguments (from auto-saved json file in inputs/json)
save_args("my_fav_args") # you can save your args; these are saved as json files in the inputs path
load_args("my_fav_args") # you can load saved args by name; these are json files in the inputs path

cls()  # clear the command window if things get cluttered
help() # display this message
exit() # exit interactive mode
"""

LAST_ARGS_PATH = DEFAULT_PATHS.inputs+"/json/last_args.json"
        
def main():
    global VERSION_STRING, INTERACTIVE_MODE_BANNER_STRING, LAST_ARGS_PATH
    global INTERACTIVE_CLI_ARGS, INTERACTIVE_CLI_STARTING_ARGS
    INTERACTIVE_CLI_ARGS = argparse.Namespace()
    
    parser = gdl.get_args_parser()
    args = parser.parse_args()
    if (args.prompt == "") and (args.interactive == False) and (args.load_args == "no_preload"):
        parser.print_help()
        exit(1)

    print("")
    if args.debug: print(VERSION_STRING + ": --debug enabled (verbose output on, writing debug files...)")
    else: print(VERSION_STRING + ": use --debug to enable verbose output and writing debug files...")
    if args.load_args != "no_preload":
        print("")
        cli_load_args(args.load_args)
        INTERACTIVE_CLI_ARGS.load_args = args.load_args
        args = INTERACTIVE_CLI_ARGS
    else:
        INTERACTIVE_CLI_ARGS = args
    
    gdl.load_pipelines(args)
    INTERACTIVE_CLI_STARTING_ARGS = argparse.Namespace(**vars(args)) # copy for reset function
    
    if args.interactive:
        cli_locals = argparse.Namespace()
        cli_locals.sample = cli_get_samples
        cli_locals.s = cli_get_samples
        cli_locals.show_args = cli_show_args
        cli_locals.sha = cli_show_args
        cli_locals.load_args = cli_load_args
        cli_locals.la = cli_load_args
        cli_locals.save_args = cli_save_args
        cli_locals.sa = cli_save_args
        cli_locals.reset_args = cli_reset_args
        cli_locals.ra = cli_reset_args
        cli_locals.cls = cli_cls
        cli_locals.help = cli_help
        cli_locals.h = cli_help
        cli_locals.e = exit
        code.interact(banner=INTERACTIVE_MODE_BANNER_STRING, local=dict(globals(), **vars(cli_locals)), exitmsg="")
        exit(0)
    else:
        if args.load_args == "no_preload": gdl.print_namespace(args, debug=args.debug, verbosity_level=1)
        
        args.init_time = str(datetime.datetime.now()) # time the command was created / queued
        samples = gdl.get_samples(args)
        gdl.save_samples(samples, args)
        
        try: # try to save the last used args in a json file for convenience
            gdl.save_json(vars(gdl.strip_args(args)), LAST_ARGS_PATH)
        except Exception as e:
            if args.debug: print("Error saving sample args - " + str(e))
            
    return
    
def cli_get_samples(prompt=None, **kwargs):
    global LAST_ARGS_PATH
    global INTERACTIVE_CLI_ARGS
    args = argparse.Namespace(**(vars(INTERACTIVE_CLI_ARGS) | kwargs)) # merge with keyword args
    if prompt: args.prompt = prompt
    if args.n < 0: # using n < 0 is the same as using repeat=True
        args.repeat = True
        args.n = 1
    if "repeat" in args: repeat = args.repeat
    else: repeat = False
    if repeat: print("Repeating sample, press ctrl+c to stop...")
    
    args.init_time = str(datetime.datetime.now()) # time the command was created / queued
    while True:
        if args.debug: importlib.reload(gdl) # this allows changes in g_diffuser_lib to take effect without restarting the cli
        
        args_copy = argparse.Namespace(**vars(args)) # preserve args, if these functions are aborted part way through
        try:                                         # anything could happen to the data
            samples = gdl.get_samples(args)
            gdl.save_samples(samples, args)
        except KeyboardInterrupt:
            print("Okay, stopping...")
            INTERACTIVE_CLI_ARGS = args_copy
            return
        
        INTERACTIVE_CLI_ARGS = args # preserve args for next call to sample()
        if args.debug: gdl.print_namespace(INTERACTIVE_CLI_ARGS, debug=0, verbosity_level=1)
        try: # try to save the last used args in a json tmp file for convenience
            gdl.save_json(vars(gdl.strip_args(args)), LAST_ARGS_PATH)
        except Exception as e:
            if args.debug: print("Error saving sample args - " + str(e))
            
        if not repeat: break
    return
    
def cli_show_args(level=None):
    global INTERACTIVE_CLI_ARGS
    if level != None: verbosity_level = level
    else: verbosity_level = 1
    gdl.print_namespace(INTERACTIVE_CLI_ARGS, debug=INTERACTIVE_CLI_ARGS.debug, verbosity_level=verbosity_level)
    return
    
def cli_load_args(name=""):
    global DEFAULT_PATHS, LAST_ARGS_PATH
    global INTERACTIVE_CLI_ARGS
    try:
        if not name: json_path = LAST_ARGS_PATH
        else: json_path = DEFAULT_PATHS.inputs+"/json/"+name+".json"
        saved_args_dict = gdl.load_json(json_path)
        INTERACTIVE_CLI_ARGS = argparse.Namespace(**(vars(INTERACTIVE_CLI_ARGS) | saved_args_dict)) # merge with keyword args
        gdl.print_namespace(INTERACTIVE_CLI_ARGS, debug=INTERACTIVE_CLI_ARGS.debug, verbosity_level=1)
    except Exception as e:
        print("Error loading last args from file - " + str(e))
    return
    
def cli_save_args(name):
    global INTERACTIVE_CLI_ARGS
    global DEFAULT_PATHS
    try:
        json_path = DEFAULT_PATHS.inputs+"/json/"+name+".json"
        saved_path = gdl.save_json(vars(gdl.strip_args(INTERACTIVE_CLI_ARGS)), json_path)
        print("Saved " + saved_path)
    except Exception as e:
        if args.debug: print("Error saving args - " + str(e))
    return

def cli_reset_args():
    global INTERACTIVE_CLI_ARGS, INTERACTIVE_CLI_STARTING_ARGS
    INTERACTIVE_CLI_ARGS = argparse.Namespace(**vars(INTERACTIVE_CLI_STARTING_ARGS))
    cli_show_args()
    return


def cli_cls():
    os.system("cls")
    return   
    
def cli_help():
    global VERSION_STRING, INTERACTIVE_MODE_BANNER_STRING
    print(VERSION_STRING+INTERACTIVE_MODE_BANNER_STRING+"\n")
    return
    
if __name__ == "__main__":
    main()