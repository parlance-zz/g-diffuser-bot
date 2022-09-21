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


g_diffuser_cli.py - command line interface for g-diffuser

"""

from g_diffuser_bot_defaults import *
import g_diffuser_lib as gdl

import os, sys
os.chdir(ROOT_PATH)

import datetime
import argparse
import code
import importlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to condition generation on",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps (number of times to refine image)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=12,
        help="guidance scale (amount of change per step)",
    )
    parser.add_argument(
        "--init-img",
        type=str,
        default="",
        help="path to the input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to the output image, if none is specified a random name will be used in the outputs folder",
        default="",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=1,
        help="use to set mask hardness ( > 0), 1 is near default hardness, lower is softer and higher is harder",
    )
    parser.add_argument(
        "--noise_q",
        type=float,
        default=1.5,
        help="augments falloff of matched noise distribution ( > 0). lower means smaller features and higher means larger features",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0,
        help="overall amount to change the input image",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
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
        "--model-name",
        type=str,
        default="",
        help="relative local path to downloaded diffusers model, or name of model if using a huggingface token",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="your huggingface developer access token (if you are using one)",
    )    
    parser.add_argument(
        "--use_optimized",
        action='store_true',
        default=False,
        help="enable memory optimizations that are currently available in diffusers",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=None,
        help="enable verbose CLI output and debug image dumps",
    )
    
    parser.add_argument(
        "--interactive",
        action='store_true',
        default=False,
        help="enters an interactive command line mode to generate multiple samples",
    )
    args = parser.parse_args()
    if (args.prompt == "") and (args.interactive == False):
        parser.print_help()
        exit(1)
    
    global DEBUG_MODE
    if args.debug != None:
        DEBUG_MODE = args.debug
        
    gdl.load_pipelines(args)
    
    if args.interactive:
        print("\nInteractive mode: call sample() with keyword args and use exit() when done, press ctrl+c to abort a repeating command:")
        print("sample('my prompt', n=3, scale=15)")
        print("sample('art by greg rutkowski', init_img='my_image_src.png', repeat=True)\n")
        print("Parameters entered as command-line arguments will be merged into your initial sample params, sample params are preserved on subsequent calls to sample()\n")
        
        if not DEBUG_MODE: print("Enable --debug for verbose output\n")
        cli_locals = argparse.Namespace()
        cli_locals.sample = _cli_get_samples
        global INTERACTIVE_CLI_ARGS
        INTERACTIVE_CLI_ARGS = args
        code.interact(local=dict(globals(), **vars(cli_locals)))
        exit(0)
    else:
        samples = gdl.get_samples(args)
        gdl.save_samples(samples, args)

def _cli_get_samples(prompt=None, **kwargs):
    global DEBUG_MODE
    global INTERACTIVE_CLI_ARGS
    args = argparse.Namespace(**gdl.merge_dicts(vars(INTERACTIVE_CLI_ARGS), kwargs))
    if prompt: args.prompt = prompt
    
    if "repeat" in args: repeat = args.repeat
    else: repeat = False
    if repeat: print("Repeating sample...")
    while True:
        if DEBUG_MODE: importlib.reload(gdl)
        samples = gdl.get_samples(args)
        gdl.save_samples(samples, args)
        INTERACTIVE_CLI_ARGS = args
        if DEBUG_MODE:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.loaded_pipes = None
            print(args_copy)
        print("")    
        if not repeat: break
    return
    
    
if __name__ == "__main__":
    main()    