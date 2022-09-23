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


g_diffuser_bot.py - discord bot interface for g-diffuser

"""

import g_diffuser_lib as gdl
from g_diffuser_config import DEFAULT_PATHS, CMD_SERVER_SETTINGS, DISCORD_BOT_SETTINGS

import os, sys
os.chdir(DEFAULT_PATHS.root)

import psutil
import pytimeparse
import pathlib
import urllib
import json
import subprocess
import glob
import asyncio
import aiohttp
import uuid
import shutil
import time
import datetime

import discord
from discord.ext import commands
from discord.ext import tasks

import numpy as np
from PIL import Image

BOT_COMMAND_LIST = ["help", "about", "examples", "gen", "queue", "cancel", "top", "select", "show_input", "shutdown", "clean", "restart", "clear", "list_servers", "leave_server"]

# help and about strings, these must be 2000 characters or less
ABOUT_TXT = """This is a simple discord bot for stable-diffusion and provides access to the most common commands as well as a few others.

Commands can be used in any channel the bot is in, provided you have the appropriate server role. For a list of commands, use !help

Please use discretion in your prompts as the safety filter has been disabled. Repeated violations will result in banning.
If you do happen to generate anything questionable please delete the message yourself or contact a mod ASAP. The watermarking feature has been left enabled to minimize potential harm.

For more information on the G-Diffuser-Bot please see https://github.com/parlance-zz/g-diffuser-bot
"""
ABOUT_TXT = ABOUT_TXT.replace("!", DISCORD_BOT_SETTINGS.cmd_prefix)

HELP_TXT1 = """
User Commands:
  !gen : Generates a new sample with the given prompt, parameters, and input attachments
  !queue : Shows running / waiting commands in the queue [-mine]
  !cancel : Cancels your last command, or optionally a specific number of commands (can be used while running) [-all]
  !top : Shows the top users' total running time
  !select : Crops an image by number from your last result and make it your input image (left to right, top to bottom) [-none]
  !show_input : Shows your current input image (skips the queue)
 
Admin Commands:
  !shutdown : Cancel all pending / running commands and shutdown the bot (can only be used by bot owner)
  !clean : Delete temporary files in SD folders, [-force] will delete temporary files that may still be referenced (can only be used by bot owner) [-force]
  !restart : Restart the bot after the command queue is idle
  !clear : Cancel all or only a specific user's pending / running commands [-all] [-user]
  !list_servers : List the names of all servers / guilds the bot is joined to
  !leave_server : Leave the specified server [-server server name]
"""
HELP_TXT1 = HELP_TXT1.replace("!", DISCORD_BOT_SETTINGS.cmd_prefix)

HELP_TXT2=""" 
Parameter Notes:
  -seed : Any whole number (default random)
  -scale : Can be any positive real number (default 10). Controls the unconditional guidance scale. Good values are between 3-20.
  -strength : (0 < strength < 1) Controls how much to change the input image.
  -steps : Any whole number from 10 to 300 (default 50). Controls how many times to iteratively refine the sample.
  -x : Repeat the given command some number of times.
  -w : Set the output width  (this will be rounded to a multiple of 64)
  -h : Set the output height (this will be rounded to a multiple of 64)
  -n : Choose the number of samples to generate at once
  -color : How much color variation to add when in/out-painting, if you use this try small values (0..1, default 0.)
  -noise_q : Controls the exponent in the in/out-painting noise distribution. Higher values means larger features and lower values means
             smaller features. (range > 0., default 1.)
  -blend : Can be used to adjust mask hardness when in/out-painting, higher values is sharper (range >= 0., default 1 (no change))
  
Examples:
  To see examples of valid commands use !examples
"""
HELP_TXT2 = HELP_TXT2.replace("!", DISCORD_BOT_SETTINGS.cmd_prefix)

EXAMPLES_TXT = """
Example commands:
!gen an astronaut riding a horse on the moon
!gen painting of an island by lisa frank -seed 10
!gen baroque painting of a mystical island treehouse on the ocean, chrono trigger, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate
!gen my little pony in space marine armor from warhammer 40k, trending on artstation, intricate detail, 3d render, gritty, dark colors, cinematic lighting, cosmic background with colorful constellations -scale 10 -seed 174468 -steps 50
!gen baroque painting of a mystical island treehouse on the ocean, chrono trigger, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate -scale 14 -seed 252229
"""
EXAMPLES_TXT = EXAMPLES_TXT.replace("!", DISCORD_BOT_SETTINGS.cmd_prefix)


"""
Input images:
  Commands that require an input image will use the image you attach to your message. If you do not attach an image it will attempt to use the last image you attached.
  The select command can be used to turn your last command's output image into your next input image, please see !select above.
"""


class Command:

    def __init__(self, ctx=None, _copy=None):
    
        self.ctx = ctx  # originating discord context is attached to command to reply appropriately
        self.args = (gdl.get_args_parser()).parse_args()   # start with default args
        self.args.interactive = True                       # commands run in a continuous session should use interactive = True
        self.args.uuid_str = gdl.get_random_string()       # attach a uuid for debugging purposes
        self.args.init_time = str(datetime.datetime.now()) # time the command was created / queued
        self.args.status = 0                               # 0 for waiting in queue, 1 for running, 2 for run successfully, 3 for cancelling, -1 for error
        
        if ctx:
            self.args.init_user = str(ctx.message.author.name)     # username that initiated the command
            self.args.message = str(ctx.message.content)           # the complete message string
            self.args.command, message_args = bot_parse_args(str(ctx.message.content))
            # todo: merge message_args into self.args
        else:
            self.args.init_user = ""
            self.args.message = ""
            
        if _copy: self.__dict__ = gdl.merge_dicts(self.__dict__, _copy.__dict__)
        return
        
    def __getstate__(self): # remove discord context for serialization
        attributes = self.__dict__.copy()
        if "ctx" in attributes:
            del attributes["ctx"]
        attributes["args"] = vars(attributes["args"]) # convert from namespace to dict for serialization
        return attributes
        
    def get_summary(self, no_init_time=False):
        args_dict = vars(self.args)
        """
        if no_init_time:
            summary = ""
        else:
            init_time = self.args.init_time.strftime("%I:%M:%S %p")
            summary = ("[" + init_time).replace("[0", "[") + "]  "
        """
        summary = ""
        
        if ("init_user" in args_dict) and ("message" in args_dict):
            summary += "@" + self.args.init_user + " " + self.args.message
        if "seed" in args_dict:
            summary += " [-seed " + str(self.args.seed) + "]"

        if self.status == 0:
            summary += " (waiting)"
        elif self.status == 1:
            summary += " (running)"
        elif self.status == 2:
            if "elapsed_time" in args_dict: summary += " (complete <" + str(self.args.elapsed_time) + "s>)"
            else: summary += " (complete)"
        elif self.status == 3:
            summary += " (cancelling)"
        elif self.status == -1:
            summary += " (error)"

        return summary
        
class CommandQueue:

    def __init__(self): #round-robin by default
        global DISCORD_BOT_SETTINGS
        global CMD_SERVER_SETTINGS
        self.settings = DISCORD_BOT_SETTINGS
        self.cmd_server_settings = CMD_SERVER_SETTINGS
        self.cmd_server_process = None
        self.cmd_list = []
        self.users_total_elapsed_time = {}
        self.restart_now = None
        
        if DISCORD_BOT_SETTINGS.state_file_path: # load existing data if we have state file path
            self.data_file_path = DISCORD_BOT_SETTINGS.state_file_path
            try:
                with open(self.data_file_path, 'r') as dfile:
                    saved_state = argparse.Namespace(**json.load(dfile))
                    dfile.close()
                print("Loaded " + self.data_file_path + "...")

                self.cmd_list = saved_state.cmd_list # we only need to restore these fields for now
                self.users_total_elapsed_time = saved_state.users_total_elapsed_time
                
            except Exception as e:
                print("Error loading '" + self.data_file_path + "' - " + str(e))
                #self.data_file_path = ""
        if not self.data_file_path: print("Warning: Discord bot state data will not be saved in this session")
        
        self.start_command_server()
        return
        
    def __getstate__(self): # remove discord context for serialization
        attributes = self.__dict__.copy()
        attributes["settings"] = vars(attributes["settings"]) # convert from namespace to dict for serialization
        attributes["cmd_server_settings"] = vars(attributes["cmd_server_settings"])
        
        cmd_list_copy = [] # only save completed commands (because we can't save contexts for waiting commands)
        for cmd in self.cmd_list:
            if cmd.args.status == 2: # completed
                cmd_list_copy.append(cmd.__getstate__())
        attributes["cmd_list"] = cmd_list_copy
        
        return attributes
        
    def save(self):
        if not self.data_file_path: return # save command queue data if we have a data file path
        try:
            pathlib.Path(self.data_file_path).touch(exist_ok=True)
            with open(self.data_file_path, "w") as dfile:
                json.dump(self, dfile)
                dfile.close()
            print("Saved " + self.data_file_path + "...")
        except Exception as e:
            print("Error saving '"+self.data_file_path+"' - "+str(e))
        return
    
    def clear(self, status_list = [0,1,3,-1]): # clears all but completed commands by default
        new_cmd_list = []
        for cmd in self.cmd_list:
            if not cmd.args.status in status_list:
                new_cmd_list.append(cmd)
        self.cmd_list = new_cmd_list
        return
        
    def get_next_pending(self, num_to_return=1, user=""):
    
        pending_list = []        
        if self.settings.queue_mode == 0: # round-robin
            user_recent_started_time = {} # find oldest queued commands per user
            for cmd in self.cmd_list:
                if cmd.args.status == 0: # queued
                    if not (cmd.args.init_user in user_recent_started_time.keys()):
                        if (user == "") or (user == cmd.args.init_user):
                            user_recent_started_time[cmd.args.init_user] = cmd.args.init_time
                        
            for cmd in self.cmd_list: # if a user has a queued command, their sort time is replaced with the most recent completed or running command start time
                if cmd.args.status in [1, 2]: # running or completed
                    if cmd.args.init_user in user_recent_started_time.keys():
                        if (user == "") or (user == cmd.args.init_user):
                            user_recent_started_time[cmd.args.init_user] = cmd.args.start_time
                            
            if len(user_recent_started_time) > 0: # are there any queued commands?
                user_last_pending_cmd_index = {}
                for user in user_recent_started_time.keys():
                    user_last_pending_cmd_index[user] = 0
                
                while len(pending_list) < num_to_return:
                    next_user = min(user_recent_started_time, key=user_recent_started_time.get)
                    next_cmd = self.get_last_command(user=next_user, status_list=[0], get_first=True, start_index=user_last_pending_cmd_index[next_user])
                    
                    if next_cmd != None:
                        user_recent_started_time[next_user] = datetime.datetime.now()
                        pending_list.append(next_cmd)
                        user_last_pending_cmd_index[next_user] = self.cmd_list.index(next_cmd) + 1
                    else:
                        break
                        
        else: # first-come first-serve
            for cmd in self.cmd_list:
                if cmd.args.status == 0: # queued
                    if (user == "") or (user == cmd.args.init_user):
                        pending_list.append(cmd)
                        if len(pending_list) >= num_to_return:
                            break
        
        if num_to_return == 1: # rather than return an empty list this method returns None or the actual command when num_to_return == 1
            if len(pending_list) == 0:
                return None
            else:
                return pending_list[0]
                
        return pending_list
        
    def get_running(self, user=""):
        for cmd in self.cmd_list:
            if cmd.args.status == 1:
                if (user == "") or (user == cmd.args.init_user):
                    return cmd # running
        return None
        
    def get_last_command(self, user=None, status_list = [0,1,2,3,-1], get_first=False, start_index=0): # by default gets the last command with any status
        if len(self.cmd_list) == 0: return None
        if get_first == False:
            search_list = self.cmd_list[::-1]
        else:
            search_list = self.cmd_list
        
        for i in range(start_index, len(self.cmd_list)):
            cmd = search_list[i]
            if (cmd.args.init_user == user) or (user == None):
                if cmd.args.status in status_list:
                    return cmd
        return None
        
    def get_last_attached(self, user=None):     
        _reversed = reversed(self.cmd_list)
        for cmd in _reversed:
            if (cmd.args.init_user == user) or (user == None):
                if cmd.args.init_img:
                    return [cmd.args.init_img]
        return []
        
    def get_last_outputs(self, user=None): # get user's last successful outputs
        _reversed = reversed(self.cmd_list)
        if user != None:
            user = user.strip().lower()
        for cmd in _reversed:
            if (cmd.args.init_user.strip().lower() == user) or (user == None):
                if (cmd.args.status == 2) and (len(cmd.args.output_samples) > 0):  # completed successfully 
                        return cmd.args.output_samples
        return [] # none found
        
    async def add_new(self, ctx): # add a new command to the queue
        rejected = (self.restart_now != None) or (self.get_queue_length() >= self.settings.max_queue_length)
        x_val = ""
        
        if rejected:
            try:
                if self.restart_now != None:
                    await ctx.send("Sorry @" + str(ctx.message.author.name) + ", please wait for restart...")
                else:
                    await ctx.send("Sorry @" + str(ctx.message.author.name) + ", the queue is too full right now...")
            except Exception as e:
                print("Error sending rejection - " + str(e))
            return

        cmd = Command(ctx)
        cmd_args_dict = vars(cmd.args)
        
        # download attachments
        in_attachments = []
        try:
            num_attachments = len(ctx.message.attachments)            
            for n in range(num_attachments):
                url = ctx.message.attachments[0].url # oof, fixed a bug that broke case-sensitive attachment URLs
                url_ext = get_file_extension_from_url(url).lower()
                if not (url_ext in self.settings.accepted_attachments):
                    print("Warning: Skipping attachment download due to invalid extension ("+url+")")
        
                attach_path = await download_attachment(url)
                if attach_path: in_attachments.append(attach_path)
        except Exception as e:
            print("Error downloading command attachments - " + str(e))
        if len(in_attachments) > 0: cmd.args.init_img = in_attachments[0]
        
        try: await ctx.send("Okay @" + cmd.args.init_user + ", gimme a sec...")
        except Exception as e: print("Error sending acknowledgement - " + str(e))
        
        if "-x" in cmd_args_dict: # make sure repeated commands dont multiply by stripping the repeat param
            x_val = cmd.args.x
            del cmd.args.x
        else: x_val = 1
        
        self.cmd_list.append(cmd)
        
        if x_val > 1: # repeat command           
            try:
                repeat_x = int(x_val) - 1
                max_repeat_limit = self.settings.repeat_limit
                if (self.settings.max_queue_length - self.get_queue_length()) < max_repeat_limit:
                    max_repeat_limit = self.settings.max_queue_length - self.get_queue_length()
                if repeat_x >= max_repeat_limit: repeat_x = max_repeat_limit - 1
                
                for x in range(repeat_x):
                    cmd_copy = Command(_copy=cmd)                    # copy the original command
                    cmd_copy.args.uuid_str = gdl.get_random_string() # but make a new guid for repeated command
                    cmd_copy.ctx = cmd.ctx                           # and keep the original discord ctx
                        
                    self.cmd_list.append(cmd_copy)
                    
            except Exception as e:
                print("Error repeating command - " + str(e))
                
        return
        
    def get_queue_str(self, max_item_count=0, user=""): # returns a string summarizing all commands in queue
    
        if max_item_count == 0: max_item_count = self.settings.max_queue_print_items
        MAX_QUEUE_STR_LENGTH = 1600
            
        i = 0
        msg = ""
        
        running_cmd = self.get_running(user=user)
        if running_cmd:
            msg += "   " + running_cmd.get_summary() + "\n"
            
        pending_list = self.get_next_pending(num_to_return=max_item_count, user=user)
        if len(pending_list) > 0:
            while len(msg) < MAX_QUEUE_STR_LENGTH:
                cmd = pending_list[i]
                i += 1
                msg += str(i) + ": " + cmd.get_summary() + "\n"
                if (i >= max_item_count) or (i >= len(pending_list)):
                    break
                
        if (i == 0) and (running_cmd == None):
            msg = "Empty!"
        
        queue_len = self.get_queue_length()
        if (len(msg) >= MAX_QUEUE_STR_LENGTH) or (i < (queue_len-1)): # truncate list if near max message length
            msg += " + " + str(self.get_queue_length() - i - 1) + " more..."
        return msg
        
    def get_queue_length(self): # returns the number of commands running or waiting in queue
        length = 0
        for cmd in self.cmd_list:
            if cmd.args.status in [0, 1]:
                length += 1
        return length

    def start_command_server(self):
        run_str = "python g_diffuser_server.py --start-server"
        self.cmd_server_process = run_string(run_str)
        return
        
    def shutdown_command_server(self):
        if self.cmd_server_process:
            _p_kill(self.cmd_server_process.pid)
            self.cmd_server_process = None
        return
        
    def restart_command_server(self):
        print("Restarting command server...")
        self.shutdown_command_server()
        self.start_command_server()
        return
        
    async def get_command_server_status(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.cmd_server_url) as response:
                    json_data = await response.json()
                    return json_data
        except Exception as e:
            print("Error getting command server status - " + str(e))
            return None
         
def get_bot_args_parser(parser=None):
    if not parser: parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to condition sampling on",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="number of sampling steps (number of times to refine image)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=11,
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
        default=1.,
        help="augments falloff of matched noise distribution for in/out-painting (noise_q > 0), lower values mean smaller features and higher means larger features",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.,
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
        help="path to downloaded diffusers model (relative to default models path), or name of model if using a huggingface token",
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
        help="preload and use a saved set of sample arguments from a json file in your inputs path",
    )
    return parser
    
def bot_parse_args(msg):

    standard_args = gdl.get_args_parser()
    standard_arg_list = []
    for action in standard_args._actions:
        for option in action.option_strings:
            if option[:2] == "--": standard_arg_list.append(option[2:]) # build a list of standard args minus the -- prefix
    
    msg_tokens = msg.replace("\t"," ").strip().split(" ") # start by tokenizing the discord msg
    command = msg_tokens[0].lower()
    arg_translation_map = { "str": "strength", "model": "model-name", "args": "load-args", "x": "repeat_x" } # translate short-hand bot args to standard args
    for i in range(1,len(msg_tokens)):
        msg_tokens[i] = msg_tokens[i].strip()
        if msg_tokens[i][:1] == "-": # only translate tokens from the discord msg that begin with a dash
            stripped_token = (msg_tokens[i].lower())[1:]        
            if stripped_token in arg_translation_map: # check the short-hand map first
                msg_tokens[i] = "--" + arg_translation_map[stripped_token]
            elif stripped_token in standard_arg_list: # but non-short-hand for standard arg names are still valid
                msg_tokens[i] = "--" + stripped_token
    msg_tokens_joined = "--prompt" + " ".join(msg_tokens[1:])
    
    PARAM_LIST = ["-str", "-scale", "-seed", "-steps", "-x", "-mine", "-all", "-num", "-force", "-user", "-w", "-h", "-n", "-none", "-color", "-noise_q", "-blend", "-server"]
    


    tokens = _shlex_split(msg)
    args = {}
    extra_args = []

    last_key = None
    for i in range(1, len(tokens)):
        if (tokens[i].lower() in param_list):
            tokens[i] = tokens[i].lower()
            args[tokens[i]] = True
            last_key = tokens[i]
        else:
            if last_key:
                args[last_key] = tokens[i]
                last_key = None
            else:
                extra_args.append(tokens[i])
    
    if len(extra_args) > 0:
        default_str = " ".join(extra_args)
        args["default_str"] = default_str
        args["prompt"] = default_str
    
        try:
            default_int = int(default_str)
            args["default_int"] = default_int
        except:
            default_int = 0
            
    return tokens[0].lower(), args

    
def get_file_extension_from_url(url):
    tokens = os.path.splitext(os.path.basename(urllib.parse.urlsplit(url).path))
    if len(tokens) > 1:
        return tokens[1]
    return ""
    
async def download_attachment(url):

    try:
        tmp_file_path = _get_tmp_path(url_ext)
        print("Downloading '" + url + "' to '" + tmp_file_path + "'...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(tmp_file_path, "wb") as out_file:
                        out_file.write(await response.read())
                        out_file.close()
                else:
                    print("Error downloading url, status = " + str(response.status))
                    return ""
        
    except Exception as e:
        print("Error downloading url - " + str(e))
        return ""
        
    return tmp_file_path
    
def _restart_program():
    global ROOT_PATH
    global CMD_QUEUE
    CMD_QUEUE.shutdown_command_server()
    time.sleep(1)
    
    SCRIPT_FILE_NAME = os.path.basename(__file__)
    
    print("Restarting...")
    run_string = 'python "' + ROOT_PATH + '/' + SCRIPT_FILE_NAME + '"'
    print(run_string)
    subprocess.Popen(run_string)
    exit(0)
    
def _auto_clean(clean_ratio=0.75):  # delete oldest images and json backups from temporary and backup paths
    global TMP_CLEAN_PATHS
    for path in TMP_CLEAN_PATHS:
        try:
            file_list = glob.glob(path)
            sorted_files = sorted(file_list, key=os.path.getctime)
            
            for i in range(int(len(sorted_files)*clean_ratio)):
                file_path = sorted_files[i]
                try:
                    print("Removing '" + file_path + "'...")
                    os.remove(file_path)
                except:
                    unable_to_remove = True
            
            print("Cleaned " + path)
        except Exception as e:
            print("Error cleaning - " + path + " - " + str(e))  
    return
    
def _check_server_roles(ctx, role_name_list): # resolve and check the roles of a user against a list of role name strings
    if ("everyone" in role_name_list):
        return True
    
    role_list = []
    for role_name in role_name_list:
        try:
            role = discord.utils.get(ctx.message.author.guild.roles, name=role_name)
            role_list.append(role)
        except:
            continue
    for role in role_list:
        if role in ctx.message.author.roles:
            return True
    return False
    
def _p_kill(proc_pid):  # kill all child processes recursively as well, its the only way to be sure
    print("Killing process - " + str(proc_pid))
    try:
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
    except Exception as e:
        print("Error killing process - " + str(proc_pid) + " - " + str(e))
    return
    
def run_string(run_string):   # run shell command asynchronously to keep discord message pumps happy and allow cancellation
    print("Running external command: " + run_string)
    try:
        process = subprocess.Popen(run_string, shell=True)
        e = ""
    except Exception as e:
        process = None
        
    if not process:
        print("Error running string '" + run_string + "' - " + str(e) + "...")
    return process
    
async def _top(ctx):    # replies to a message with a sorted list of all users and their run-time
    global CMD_QUEUE
    msg = "Okay @" + ctx.message.author.name + ", here's the top users... \n"
    i = 0
    for user in sorted(CMD_QUEUE.users_elapsed_time, reverse=True, key=CMD_QUEUE.users_elapsed_time.get):
        i += 1
        msg += str(i) + ": @" + user + " <" + str(datetime.timedelta(seconds=CMD_QUEUE.users_elapsed_time[user].seconds)) + "s>\n"
    if i == 0:
        msg = "No users yet!"
    await ctx.send("@" + ctx.message.author.name + " : " + msg)
    
async def _select(ctx): # crop an image from the user's last output image grid
    global CMD_QUEUE
    command, cmd_args = bot_parse_args(ctx.message.content)
    
    output_attachments = []
    try:
        user = ctx.message.author.name
        author = user
        
        if "-user" in cmd_args:
            user = cmd_args["-user"]
        
        select_num = _get_int_arg("-num", cmd_args)
        if select_num == None: select_num = 1
            
        output_attachments = CMD_QUEUE.get_last_outputs(user=user)
        if len(output_attachments) == 0:
            raise Exception("No output images to select")
            
        output_attachments = [output_attachments[select_num-1]]
            
        _reversed = reversed(CMD_QUEUE.cmd_list) # look for the most recently completed command by the requesting user with an attached image and replace it with the selected one
        for cmd in _reversed:
            if cmd.args.init_user == author:
                if cmd.args.status == 2: # completed successfully
                    cmd.in_attachments = output_attachments
                    break
        
    except Exception as e:
        await ctx.send("Sorry @" + ctx.message.author.name + ", " + str(e))
    
    return output_attachments

if __name__ == "__main__":

    # if we don't actually have a discord bot token let's not go any further
    if (not DISCORD_BOT_SETTINGS.token) or (DISCORD_BOT_SETTINGS.token == "YOUR_DISCORD_BOT_TOKEN_HERE"):
        print("Fatal error: Cannot start discord bot with token '" + DISCORD_BOT_SETTINGS.token + "'")
        print("Please update DISCORD_BOT_SETTINGS.token in g_diffuser_config.py and run this again.")
        exit(1)
        
    intents = discord.Intents().default() # this bot requires both message and message content intents (message content is a privileged intent)
    intents.messages = True
    intents.dm_messages = True
    intents.message_content = True

    client = commands.Bot(command_prefix=DISCORD_BOT_SETTINGS.cmd_prefix, intents=intents)
    client.remove_command('help') # required to make the custom help command work
    
@client.event
async def on_ready():
    global DISCORD_BOT_SETTINGS
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=DISCORD_BOT_SETTINGS.activity))
    _process_commands_loop.start()
    return
 
@client.command()
@commands.is_owner()
async def shutdown(ctx): # shutdown the bot (only used by the bot owner)
    global CMD_QUEUE
    CMD_QUEUE.shutdown_command_server()
    await ctx.send("Bye")
    exit()
    
@client.command()
async def restart(ctx): # restart the bot when the queue is empty (available to admins)
                        # to restart immediately an admin can follow-up with !clear
    global CMD_QUEUE
    command, cmd_args = bot_parse_args(ctx.message.content)
    
    global BOT_ADMIN_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME]): return
        
    CMD_QUEUE.restart_now = ctx
    
    if "-force" in cmd_args:
        for cmd in CMD_QUEUE.cmd_list: 
            if cmd.args.status in [0, 1]: # cancel everything running or waiting
                cmd.args.status = 3 # cancelled
        
        try:
            await ctx.send("Okay @" + ctx.message.author.name + ", clearing queue and restarting now... ")
        except Exception as e:
            print("Error sending restart acknowledgement - " + str(e))
    else:
        try:
            await ctx.send("Okay @" + ctx.message.author.name + ", restarting when queue is empty... ")
        except Exception as e:
            print("Error sending restart acknowledgement - " + str(e))
        
    return

@client.command()
@commands.is_owner()
async def leave_server(ctx):        # leave a server / guild that the bot is joined to (owner only)
    command, cmd_args = bot_parse_args(ctx.message.content)
    if "-server" in cmd_args:
        server_name = str(cmd_args["-server"])
        if "default_str" in cmd_args:
            server_name += " " + "".join(cmd_args["default_str"])
        try:
            server = None
            server = discord.utils.get(client.guilds, name=server_name) # Get the server / guild by name
        except:
            print("Error retrieving server object in !leave_server")
            
        if server is None:
            try:
                await ctx.send("Sorry @" + ctx.message.author.name + ", I'm not on '" + server_name + "'... ")
            except Exception as e:
                print("Error sending leave_server acknowledgement - " + str(e))
            return
    else:
        try:
            await ctx.send("Sorry @" + ctx.message.author.name + ", try !leave_server -server [server_name]... ")
        except Exception as e:
            print("Error sending leave_server acknowledgement - " + str(e))
        return
            
    try:
        await ctx.send("Okay @" + ctx.message.author.name + ", leaving '" + server_name + "'... ")
    except Exception as e:
        print("Error sending leave_server acknowledgement - " + str(e))

    try:
        await server.leave()
    except Exception as e:
        print("Error leaving server '" + server_name + "' - " + str(e))
        
@client.command()
@commands.is_owner()
async def list_servers(ctx):        # list servers / guilds bot is a member of (owner only)
    
    msg = "Okay @" + ctx.message.author.name + ", here's the servers I'm on... \n"
    for server in client.guilds:
        msg += server.name + "\n"
        
    try:
        await ctx.send(msg)
    except Exception as e:
        print("Error sending list_servers - " + str(e))
    
    return
    
@client.command()
async def clear(ctx): # clear the command queue completely (available to admins)
    
    global CMD_QUEUE
    
    global BOT_ADMIN_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME]): return
    command, cmd_args = bot_parse_args(ctx.message.content)
    
    user = ""
    if "-all" in cmd_args: # clear the whole queue
        try:
            user = "*"
            await ctx.send("Okay @" + ctx.message.author.name + ", clearing the queue... ")
        except:
            print("Error sending acknowledgement")
    elif "-user" in cmd_args: # clear the requested user's queue
        try:
            user = cmd_args["-user"]
            await ctx.send("Okay @" + ctx.message.author.name + ", clearing @" + user + " from the queue... ")
        except:
            print("Error sending acknowledgement")

    if user == "":
        try:
            await ctx.send("Sorry @" + ctx.message.author.name + ", please use !clear -all or !clear -user [user]... ")
        except:
            print("Error sending acknowledgement")
        return
        
    for cmd in CMD_QUEUE.cmd_list: # do the cancelling
        if cmd.args.status in [0, 1]:
            if user == "*":
                cmd.args.status = 3 # cancelled
            else:
                if cmd.args.init_user.strip().lower() == user.strip().lower():
                    cmd.args.status = 3 # cancelled
        
    return

@client.command()
@commands.is_owner()
async def clean(ctx): # clean all temp folders (only used by the bot owner)
    command, cmd_args = bot_parse_args(ctx.message.content)
    
    if "-force" in cmd_args:
        _auto_clean(clean_ratio=1.0)
        await ctx.send("Okay @" + ctx.message.author.name + ", cleaning ALL temp files... ")
    else:
        _auto_clean()
        await ctx.send("Okay @" + ctx.message.author.name + ", cleaning temp files... ")

    return
    
@client.command()
async def cancel(ctx): # stops the requesting user's last queued command, or all of them
    global CMD_QUEUE
    command, cmd_args = bot_parse_args(ctx.message.content)
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    
    if "-all" in cmd_args:
        for cmd in CMD_QUEUE.cmd_list:
            if cmd.args.init_user == ctx.message.author.name:
                if cmd.args.status in [0, 1]: # queued or in-progress
                    cmd.args.status = 3 # cancelled

        await ctx.send("Okay @" + ctx.message.author.name + ", cancelling all your queued commands...")
        return
        
    else:
    
        num_to_cancel = _get_int_arg("-x", cmd_args)
        if num_to_cancel == None: num_to_cancel = 1
        
        num_cancelled = 0
        for i in range(num_to_cancel):

            cmd = CMD_QUEUE.get_last_command(user=ctx.message.author.name, status_list=[0,1])   # queued or in-progress
            if cmd:
                cmd.args.status = 3 # cancelled
                num_cancelled += 1
            else:
                break
        
        if num_cancelled > 0: 
            await ctx.send("Okay @" + ctx.message.author.name + ", cancelling your last " + str(num_cancelled) + " command(s)")
        else:
            await ctx.send("Sorry @" + ctx.message.author.name + ", no running or waiting commands to cancel...")
    
@client.command()
async def show_input(ctx, attachments=None): # attaches the requesting user's input image in response, or the images in attachments if not None

    global CMD_QUEUE
    
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    
    if attachments:
        input_paths = attachments
    else:
        input_paths = CMD_QUEUE.get_last_attached(user=ctx.message.author.name)
        
    if len(input_paths) > 0:
        try:
            files = []
            for input_path in input_paths:
                file = discord.File(input_path)
                files.append(file)
            msg = "@" + ctx.message.author.name + ":"
            await ctx.send(files=files, content=msg)
        except Exception as e:
            print("Error sending show user input image - " + str(e))
            try:
                await ctx.send("Sorry @" + ctx.message.author.name + ", I can't find that image...")
            except:
                print("")
        
    else:
        try:
            await ctx.send("Sorry @" + ctx.message.author.name + ", no attachments to show")
        except Exception as e:
            print("Error sending show user input rejection - " + str(e))
    return
    
@client.command()
async def hello(ctx):
    await ctx.send("Hi")
    
@client.command() # show the next part of the pending command list
async def queue(ctx):
    global CMD_QUEUE
    command, cmd_args = bot_parse_args(ctx.message.content)
    
    if "-mine" in cmd_args:
        msg = "Okay @" + ctx.message.author.name + ", here's your queue... \n"
        msg += CMD_QUEUE.get_queue_str(user=ctx.message.author.name)
    else:
        msg = "Okay @" + ctx.message.author.name + ", here's the queue... \n"
        msg += CMD_QUEUE.get_queue_str()
            
    await ctx.send(msg)

@client.command()
async def top(ctx):
    await _top(ctx)

@client.command()
async def scoreboard(ctx):
    await _top(ctx)

@client.command()
async def select(ctx):
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return
    attachments = await _select(ctx)
    await show_input(ctx, attachments=attachments)
    return
       
@tasks.loop(seconds = DISCORD_BOT_SETTINGS.queue_poll_interval) # repeat at polling interval
async def _process_commands_loop():

    global CMD_QUEUE
    global DEFAULT_CMD_PARAMS
    global AUTO_SEED_RANGE
    global CMD_SERVER_BIND_HTTP_HOST
    global CMD_SERVER_BIND_HTTP_PORT

    if CMD_QUEUE.get_running(): return # don't re-enter if something is already running
    CMD_QUEUE.clear(status_list = [3]) # clear cancelled
    cmd = CMD_QUEUE.get_next_pending()
    
    if cmd == None:
        if CMD_QUEUE.restart_now:
            print("Restarting now...")
            try:
                await CMD_QUEUE.restart_now.send("Restarting now...")
            except:
                print("Error sending restart message")
            try:
                await asyncio.sleep(2)
                _restart_program()
            except:
                exit(1)
                
        return # nothing to do
    
    cmd.args.status = 1 # running
    
    # don't think I'll actually this, it reduces performance so disabling for now
    #command_server_status = await CMD_QUEUE.get_command_server_status()
    #if not command_server_status:           # verify the command server is running and responsive
    #    CMD_QUEUE.restart_command_server()  # if it isn't, forcibly kill the whole process and restart it

    auto_seed = str(np.random.randint(AUTO_SEED_RANGE[0], AUTO_SEED_RANGE[1]))  # create an auto seed
    default_params = DEFAULT_CMD_PARAMS.copy()
    default_params["-seed"] = auto_seed
    cmd.args = _merge_dicts(default_params, cmd.args) # add default params
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CMD_QUEUE.cmd_server_url, json=cmd.__getstate__()) as response:
                json_data = await response.json()

    except Exception as e:
        json_data = None
        cmd.args.status = -1 # error status
        cmd.error_txt = "Error sending command to command server - " + str(e)
        
    if cmd.args.status == 3: # cancelled status
        return          # silently return
        
    if json_data: # copy result attributes back from the command server if the command wasn't cancelled while running
        try:
            attribs = { "status": int,
                        "start_time": datetime.datetime,
                        "error_txt": str,
                        "elapsed_time": datetime.timedelta,
                        "out_attachments": list,
                        "out_resolution": tuple,
                        "out_preview_image": str,
                        "out_preview_image_layout": tuple }
            _set_attribs_from_json(cmd, attribs, json_data)

        except Exception as e:
            cmd.args.status = -1 # error status
            cmd.error_txt = "Error parsing results from command server - "  + str(e)
        
    next_cmd = CMD_QUEUE.get_next_pending()
    
    if cmd.args.status == 2: # completed successfully
    
        # update the per user total run-time cache
        if cmd.args.init_user in CMD_QUEUE.users_elapsed_time.keys():
            CMD_QUEUE.users_elapsed_time[cmd.args.init_user] += cmd.elapsed_time
        else:
            CMD_QUEUE.users_elapsed_time[cmd.args.init_user] = cmd.elapsed_time

        # send requesting user their output attachments
        try:
            files = []
            if cmd.out_preview_image != "": # use summary / preview image if available and send only 1 attachment
                files.append(discord.File(cmd.out_preview_image))
            else:
                for out_file in cmd.args.output_samples:
                    files.append(discord.File(out_file))
                    
            msg = "Finished " + cmd.get_summary(no_init_time=True)
            if next_cmd == None: msg += " - Queue is empty!"
            if cmd.error_txt != "": msg += "\n" + cmd.error_txt
            
            await cmd.ctx.send(files=files, content=msg)
        except Exception as e:
            cmd.args.status = -1 # error
            cmd.error_txt = "Error sending output image - " + str(e)
            
    if cmd.args.status == -1: # error
        
        try:
            print("Error processing command - " + cmd.error_txt)
            msg = "Sorry @" + cmd.args.init_user + ", " + cmd.error_txt + " - " + cmd.get_summary()
            if next_cmd == None: msg += " - Queue is empty!"
            await cmd.ctx.send(msg)
        except Exception as e:
            print("Error sending error message to user - " + str(e))
    
    # write command file to disk if we have a data file path
    
    CMD_QUEUE.save()
    
    return

@client.command()
async def gen(ctx):
    global CMD_QUEUE
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    await CMD_QUEUE.add_new(ctx)
    
@client.command()
async def enhance(ctx):
    global CMD_QUEUE
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return
    await CMD_QUEUE.add_new(ctx)

@client.command()
async def about(ctx):
    global ABOUT_TXT
    await ctx.send("@" + ctx.message.author.name + " : " + ABOUT_TXT)
    
@client.command()
async def help(ctx):
    global HELP_TXT1, HELP_TXT2
    await ctx.send("@" + ctx.message.author.name + " : " + HELP_TXT1)
    await ctx.send("@" + ctx.message.author.name + " : " + HELP_TXT2)
    
@client.command()
async def examples(ctx):
    global EXAMPLES_TXT
    await ctx.send("@" + ctx.message.author.name + " : " + EXAMPLES_TXT)
    
@client.event
async def on_message(message):
    global DISCORD_BOT_SETTINGS
    global BOT_COMMAND_LIST
    
    if message.author.bot: return
    if message.content.startswith(DISCORD_BOT_SETTINGS.cmd_prefix) != True: return
    
    command, cmd_args = bot_parse_args(message.content)
    if command.lower().replace(cmd_prefix, "") not in BOT_COMMAND_LIST: return
    
    await client.process_commands(message)

@client.event
async def on_disconnect():
    await asyncio.sleep(10)

if __name__ == "__main__":
        
    CMD_QUEUE = CommandQueue()
    client.run(DISCORD_BOT_SETTINGS.token, reconnect=True)