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

import g_diffuser_lib as gdl
from g_diffuser_config import DEFAULT_PATHS

import os, sys
os.chdir(DEFAULT_PATHS.root)

import datetime
import argparse
import code
import importlib
  
VERSION_STRING = "g-diffuser-cli v0.4"

def main():
    global VERSION_STRING
    global INTERACTIVE_CLI_ARGS
    INTERACTIVE_CLI_ARGS = argparse.Namespace()
    
    parser = gdl.get_args_parser()
    args = parser.parse_args()    
    if (args.prompt == "") and (args.interactive == False) and (args.load_args == "no_preload"):
        parser.print_help()
        exit(1)

    if args.debug: print(VERSION_STRING + ": --debug enabled (verbose output on, writing debug file dumps to tmp...)")
    else: print(VERSION_STRING + ": use --debug to enable verbose output and writing debug files to tmp...")
    if args.load_args != "no_preload":
        print("")
        cli_load_args(args.load_args)
        INTERACTIVE_CLI_ARGS.load_args = args.load_args
        args = INTERACTIVE_CLI_ARGS
    else:
        INTERACTIVE_CLI_ARGS = args
        
    gdl.load_pipelines(args)
    
    if args.interactive:
        print("\nInteractive mode: call sample() with keyword args and use exit() when done:")
        print("sample('my prompt', n=3, scale=15)")
        print("sample('art by greg rutkowski', init_img='my_image_src.png', repeat=True, debug=True)")
        print("sample()    # sample() can be used without any args to use your last args instead")
        print("show_args() # shows the current input/output arguments")
        print("load_args() # load the last used args (from auto-save file)")
        print("load_args('my_fav_args') # you can load a named arg file")
        print("save_args('my_fav_args') # you can save a named arg file")
        print("cls() # clear the command window if things get cluttered\n")
        print("Current args: " + str(gdl.strip_args(args))+"\n")
        
        cli_locals = argparse.Namespace()
        cli_locals.sample = cli_get_samples
        cli_locals.show_args = cli_show_args
        cli_locals.load_args = cli_load_args
        cli_locals.load_last_args = cli_load_args
        cli_locals.save_args = cli_save_args
        cli_locals.cls = cli_cls
        code.interact(local=dict(globals(), **vars(cli_locals)))
        exit(0)
    else:
        if args.load_args == "no_preload": print("Sample arguments: " + str(gdl.strip_args(args))+"\n")
        
        args.init_time = str(datetime.datetime.now()) # time the command was created / queued
        samples = gdl.get_samples(args)
        gdl.save_samples(samples, args)
        
        try: # try to save the last used args in a json file for convenience
            gdl.save_json(vars(gdl.strip_args(args)), DEFAULT_PATHS.inputs+"/last_args.json")
        except Exception as e:
            if args.debug: print("Error saving sample args - " + str(e))
            
    return
    
def cli_get_samples(prompt=None, **kwargs):
    global INTERACTIVE_CLI_ARGS
    
    args = argparse.Namespace(**gdl.merge_dicts(vars(INTERACTIVE_CLI_ARGS), kwargs))
    if prompt: args.prompt = prompt
    if args.n < 0:
        args.repeat = True
        args.n = 1    
    if "repeat" in args: repeat = args.repeat
    else: repeat = False
    if repeat: print("Repeating sample, press ctrl+c to stop...")
    
    args.init_time = str(datetime.datetime.now()) # time the command was created / queued
    while True:
        importlib.reload(gdl) # this allows changes in g_diffuser_lib to take effect without restarting the cli
        
        samples = gdl.get_samples(args)
        gdl.save_samples(samples, args)
        
        INTERACTIVE_CLI_ARGS = args # preserve args for next call to sample()
        if args.debug: print(str(gdl.strip_args(args))+"\n")
        try: # try to save the last used args in a json tmp file for convenience
            gdl.save_json(vars(gdl.strip_args(args)), DEFAULT_PATHS.inputs+"/last_args.json")
        except Exception as e:
            if args.debug: print("Error saving sample args - " + str(e))
            
        if not repeat: break
    return
    
def cli_show_args():
    global INTERACTIVE_CLI_ARGS
    print("Current args: " + str(gdl.strip_args(INTERACTIVE_CLI_ARGS))+"\n")
    return
    
def cli_load_args(name=""):
    global INTERACTIVE_CLI_ARGS
    try:
        if not name: name = "last_sample_args"
        saved_args_dict = gdl.load_json(name)
        INTERACTIVE_CLI_ARGS = argparse.Namespace(**gdl.merge_dicts(vars(INTERACTIVE_CLI_ARGS), saved_args_dict))
        print("Loaded args from file: " + str(gdl.strip_args(INTERACTIVE_CLI_ARGS))+"\n")
    except Exception as e:
        print("Error loading last args from file - " + str(e))
    return
    
def cli_save_args(name):
    global INTERACTIVE_CLI_ARGS
    global DEFAULT_PATHS
    try:
        saved_path = gdl.save_json(vars(gdl.strip_args(INTERACTIVE_CLI_ARGS)), DEFAULT_PATHS.inputs+"/"+name+".json")
        print("Saved " + saved_path)
    except Exception as e:
        if args.debug: print("Error saving args - " + str(e))
    return

def cli_cls():
    os.system("cls")
    return   
    
if __name__ == "__main__":
    main()