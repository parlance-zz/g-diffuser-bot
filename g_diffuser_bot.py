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

from g_diffuser_bot_defaults import *
import g_diffuser_lib as gdl

import os, sys
os.chdir(ROOT_PATH)

# -------------------------------------------------------------------------------------------

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
import shlex
import time
import datetime

import discord
from discord.ext import commands
from discord.ext import tasks

import numpy as np
from PIL import Image


class Command:

    def __init__(self, ctx=None, _copy=None):
        global PARAM_LIST
    
        self._id = str(uuid.uuid4())
        self.ctx = ctx  # originating discord context is attached to command to reply appropriately
        self.init_time = datetime.datetime.now() # time the command was created / queued
        if ctx:
            self.init_user = ctx.message.author.name                        # username that initiated the command
            self.message = ctx.message.content                              # the complete message string
            self.command, self.cmd_args = _parse_args(ctx.message.content, PARAM_LIST) # string and dictionary
        else:
            self.init_user = ""
            self.message = ""
            self.command = ""
            self.cmd_args = []
            
        self.in_attachments = []             # local path to input attachments
        self.out_attachments = []            # local path to output attachments
        self.out_resolution = (0, 0)         # size of the individual images / movie resolution in the output
        self.out_audio_resolution = (0, 0)   # sample rate and num_channels of any generated audio in the output
        
        self.out_preview_image = ""            # if multiple outputs were generated this can be a path to a composite grid
        self.out_preview_image_layout = (1, 1) # grid layout of preview image, or (1,1) for a single image
        
        self.start_time = datetime.datetime.max             # time when the command begins running
        self.elapsed_time = datetime.timedelta(seconds=0)   # if the command completed successfully this is the run-time
        self.status = 0         # 0 for waiting in queue, 1 for running, 2 for run successfully, 3 for cancelling, -1 for error
        self.error_txt = ""     # if the command had an error, this text is any additional info
    
        if _copy:
            attribs = { "status": int,
                        "init_time": datetime.datetime,
                        "start_time": datetime.datetime,
                        "init_user": str,
                        "message": str,
                        "command": str,
                        "cmd_args": dict,
                        "error_txt": str,
                        "elapsed_time": datetime.timedelta,
                        "in_attachments": list,
                        "out_attachments": list,
                        "out_resolution": tuple,
                        "out_audio_resolution": tuple,
                        "out_preview_image": str,
                        "out_preview_image_layout": tuple }
            _set_attribs_from_json(self, attribs, _copy)
                    
    def __getstate__(self):      # remove discord context for pickling
        attributes = self.__dict__.copy()
        if "ctx" in attributes.keys():
            del attributes["ctx"]
            
        # datetime hates json and json hates datetime
        attributes["init_time"] = str(attributes["init_time"])
        attributes["start_time"] = str(attributes["start_time"])
        attributes["elapsed_time"] = str(attributes["elapsed_time"])
        return attributes
        
    def get_summary(self, no_init_time=False):
        if no_init_time:
            summary = ""
        else:
            summary = ("[" + self.init_time.strftime("%I:%M:%S %p")).replace("[0", "[") + "]  "
        summary += "@" + self.init_user + " " + self.message
        
        if "-seed" in self.cmd_args:
            summary += " [-seed " + self.cmd_args["-seed"] + "]"
            
        if self.status == 0:
            summary += " (waiting)"
        elif self.status == 1:
            summary += " (running)"
        elif self.status == 2:
            summary += " (complete <" + str(datetime.timedelta(seconds=self.elapsed_time.seconds)) + "s>)"
        elif self.status == 3:
            summary += " (cancelling)"
        elif self.status == -1:
            summary += " (error)"
        return summary
        
class CommandQueue:

    def __init__(self, queue_mode=0): #round-robin by default
        global ROOT_PATH
        global BACKUP_PATH
        global BOT_STATE_DATA_FILE
        global CMD_SERVER_BIND_HTTP_HOST
        global CMD_SERVER_BIND_HTTP_PORT
        
        data_file = BOT_STATE_DATA_FILE
        
        self.cmd_list = []
        self.data_file_path = ""
        self.users_elapsed_time = {}
        self.restart_now = None
        self.queue_mode = queue_mode
        
        if data_file != "": # load existing data if given a data file path
            self.data_file_path = data_file
            try:
                # try to make sure backup folder exists
                pathlib.Path(BACKUP_PATH).mkdir(exist_ok=True)

                file_size = os.stat(data_file).st_size # if the file is empty, look for the last backup instead
                if file_size == 0:
                    list_of_files = glob.glob(BACKUP_PATH + "/*.json")
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print("'" + data_file + "' is empty, loading '" + latest_file + "' instead...")
                    self.data_file_path = latest_file
                    data_file = latest_file
                
                with open(data_file, 'r') as dfile:
                    cmd_list_copy = json.load(dfile)
                    dfile.close()
                
                for _cmd in cmd_list_copy:
                    cmd = Command(_copy=_cmd)
                    self.cmd_list.append(cmd)
                    
                self.recalculate_user_times()
                print("Loaded " + data_file + "...")
                
            except Exception as e:
                print("Error loading '" + data_file + "' - " + str(e))
    
        self.cmd_server_url = "http://" + CMD_SERVER_BIND_HTTP_HOST + ":" + str(CMD_SERVER_BIND_HTTP_PORT)
        self.command_server_process = None
        self.start_command_server()
        
        return
    
    def recalculate_user_times(self): # (re)create the per user total run-time cache
        self.users_elapsed_time = {}
        for cmd in self.cmd_list:
            if cmd.init_user in self.users_elapsed_time.keys():
                self.users_elapsed_time[cmd.init_user] += cmd.elapsed_time
            else:
                self.users_elapsed_time[cmd.init_user] = cmd.elapsed_time
                
    def save(self):
        global BACKUP_PATH
        if self.data_file_path != "": # save command queue data if we have a data file path
            try:
                file = Path(self.data_file_path)
                file.touch(exist_ok=True)
                
                old_file_size = os.stat(self.data_file_path).st_size
                if old_file_size > 0:
                    backup_path = BACKUP_PATH + "/" + str(uuid.uuid4()) + ".json"
                    file = Path(backup_path)
                    file.touch()
                    shutil.copyfile(self.data_file_path, backup_path) # copy to make a backup file (if not empty)
                    print("Backed up data file to " + backup_path + "...")
                
                cmd_list_copy = []
                
                for cmd in self.cmd_list:
                    if cmd.status == 2: # only save completed commands (because we can't save contexts for waiting commands)
                        cmd_list_copy.append(cmd.__getstate__())
                
                with open(self.data_file_path, "w") as file:
                    json.dump(cmd_list_copy, file)
                    file.close()
                
                print("Saved " + self.data_file_path + "...")
                
            except Exception as e:
                print("Error saving " + self.data_file_path + " (" + str(e) + ")")
    
    def clear(self, status_list = [0,1,3,-1]): # clears all but completed commands by default
        new_cmd_list = []
        for cmd in self.cmd_list:
            if not cmd.status in status_list:
                new_cmd_list.append(cmd)
        
        self.cmd_list = new_cmd_list
        return
    
    def compactify(self): # reduces completed commands for each user to a single summary command
        #todo:
        return
        
    def get_next_pending(self, num_to_return=1, user=""):
    
        pending_list = []        
        if self.queue_mode == 0: # round-robin
            user_recent_started_time = {} # find oldest queued commands per user
            for cmd in self.cmd_list:
                if cmd.status == 0: # queued
                    if not (cmd.init_user in user_recent_started_time.keys()):
                        if (user == "") or (user == cmd.init_user):
                            user_recent_started_time[cmd.init_user] = cmd.init_time
                        
            for cmd in self.cmd_list: # if a user has a queued command, their sort time is replaced with the most recent completed or running command start time
                if cmd.status in [1, 2]: # running or completed
                    if cmd.init_user in user_recent_started_time.keys():
                        if (user == "") or (user == cmd.init_user):
                            user_recent_started_time[cmd.init_user] = cmd.start_time
                            
            if len(user_recent_started_time) > 0: # are there any queued commands?
                user_last_pending_cmd_index = {}
                for user in user_recent_started_time.keys():
                    user_last_pending_cmd_index[user] = 0
                
                while len(pending_list) < num_to_return:
                    next_user = min(user_recent_started_time, key=user_recent_started_time.get)
                    next_cmd = self.get_last_command(user=next_user, status_list=[0], get_first=True, start_index=user_last_pending_cmd_index[next_user])
                    
                    if next_cmd != None:
                        user_recent_started_time[next_user] = datetime.datetime.now()
                        time.sleep(0.005)
                        pending_list.append(next_cmd)
                        user_last_pending_cmd_index[next_user] = self.cmd_list.index(next_cmd) + 1
                    else:
                        break
                        
        else: # first-come first-serve
            for cmd in self.cmd_list:
                if cmd.status == 0: # queued
                    if (user == "") or (user == cmd.init_user):
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
            if cmd.status == 1:
                if (user == "") or (user == cmd.init_user):
                    return cmd # running
        return None
        
    def get_last_command(self, user=None, status_list = [0,1,2,3,-1], get_first=False, start_index=0): # by default gets the last command with any status
        if len(self.cmd_list) == 0: return None
        if user == None: return self.cmd_list[-1]
        
        if get_first == False:
            search_list = self.cmd_list[::-1]
        else:
            search_list = self.cmd_list
        
        for i in range(start_index, len(self.cmd_list)):
            cmd = search_list[i]
            if (cmd.init_user == user) or (user == None):
                if cmd.status in status_list:
                    return cmd
                
        return None
        
    def get_last_attached(self, user=None):     
        _reversed = reversed(self.cmd_list)
        for cmd in _reversed:
            if (cmd.init_user == user) or (user == None):
                if len(cmd.in_attachments) > 0:
                    return cmd.in_attachments
                    
        return []
        
    def get_last_output(self, user=None): # get user's last successful output
        attachments = []
        _reversed = reversed(self.cmd_list)
        if user != None:
            user = user.strip().lower()
        for cmd in _reversed:
            if (cmd.init_user.strip().lower() == user) or (user == None):
                if (cmd.status == 2) and (len(cmd.out_attachments) > 0):  # completed successfully
                        return cmd.out_attachments
        return attachments        
        
    async def add_new(self, ctx): # add a new command to the queue
        global MAX_REPEAT_LIMIT
        global MAX_QUEUE_LENGTH

        rejected = (self.restart_now != None) or (self.get_queue_length() >= MAX_QUEUE_LENGTH)
        x_val = ""
        
        if rejected:
            try:
                if self.restart_now != None:
                    await ctx.send("Sorry @" + ctx.message.author.name + ", please wait for restart...")
                else:
                    await ctx.send("Sorry @" + ctx.message.author.name + ", the queue is too full right now...")
            except Exception as e:
                print("Error sending rejection - " + str(e))
            return
                
        cmd = Command(ctx)
        
        if "-x" in cmd.cmd_args.keys(): # make sure repeated commands dont multiply by stripping the repeat param
            x_val = cmd.cmd_args["-x"]
            del cmd.cmd_args["-x"]
        
        # download attachments
        try:
            num_attachments = len(ctx.message.attachments)            
            for n in range(num_attachments):
                url = ctx.message.attachments[0].url.lower()
                attach_path = await _download_to_tmp(url)
                if attach_path != "":
                    cmd.in_attachments.append(attach_path)
                
        except Exception as e:
            print("Error downloading command attachment - " + str(e))
        
        try:
            await ctx.send("Okay @" + cmd.init_user + ", gimme a sec...")
        except Exception as e:
            print("Error sending acknowledgement - " + str(e))
        
        self.cmd_list.append(cmd)
        
        if x_val != "": # repeat command           
            try:
                repeat_x = int(x_val) - 1
                max_repeat_limit = MAX_REPEAT_LIMIT
                if (MAX_QUEUE_LENGTH - self.get_queue_length()) < max_repeat_limit:
                    max_repeat_limit = MAX_QUEUE_LENGTH - self.get_queue_length()
                    
                if repeat_x >= max_repeat_limit:
                    repeat_x = max_repeat_limit - 1
                
                for x in range(repeat_x):
                    cmd_copy = Command(_copy=cmd.__getstate__())
                    cmd_copy._id = str(uuid.uuid4()) # make a new guid for repeated command
                    cmd_copy.ctx = cmd.ctx           # but keep the discord ctx
                    
                    if "-x" in cmd_copy.cmd_args:
                        del cmd_copy.cmd_args["-x"] # make sure repeats don't multiply ;)
                        
                    self.cmd_list.append(cmd_copy)
                    
            except Exception as e:
                print("Error repeating command - " + str(e))
                
        return
        
    def get_queue_str(self, max_item_count=10, user=""): # returns a string summarizing all commands in queue
        global MAX_QUEUE_PRINT_ITEMS    
        MAX_QUEUE_STR_LENGTH = 1600
        
        max_item_count = MAX_QUEUE_PRINT_ITEMS
        
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
            if cmd.status in [0, 1]:
                length += 1
        return length

    def start_command_server(self):
        run_string = "python g_diffuser_server.py --start_server"
        self.command_server_process = _run_string(run_string)
        return
        
    def shutdown_command_server(self):
        if self.command_server_process:
            _p_kill(self.command_server_process.pid)
            self.command_server_process = None
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
    
def _set_attribs_from_json(obj, attribs, json_data):
    for attrib in attribs.keys():
        _class = attribs[attrib]
        _val = json_data[attrib]
        val = None
        if _class == datetime.datetime:
            val = datetime.datetime.strptime(_val.split(".")[0], "%Y-%m-%d %H:%M:%S")
        elif _class == datetime.timedelta:
            val = datetime.timedelta(seconds=pytimeparse.parse(_val))
        elif _class == tuple:
            val = tuple(_val)
        else:
            val = _val
            
        if val:
            setattr(obj, attrib, val)
    
    return
        
    cmd.status = _cmd["status"]
    cmd.start_time = datetime.datetime.strptime(_cmd["start_time"].split(".")[0], "%Y-%m-%d %H:%M:%S")
    cmd.error_txt = _cmd["error_txt"]
    cmd.elapsed_time = datetime.timedelta(seconds=pytimeparse.parse(_cmd["elapsed_time"]))
    cmd.out_attachments = _cmd["out_attachments"]
    cmd.out_resolution = (_cmd["out_resolution"][0], _cmd["out_resolution"][1])
    cmd.out_preview_image = _cmd["out_preview_image"]
    cmd.out_preview_image_layout = (_cmd["out_preview_image_layout"][0], _cmd["out_preview_image_layout"][1])
            
def _shlex_split(value):    #  did this to make things easier on windows
    lex = shlex.shlex(value)
    lex.quotes = '"'
    lex.whitespace_split = True
    lex.commenters = ''
    return list(lex)
    
def _parse_args(msg, param_list):
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

def _get_int_arg(name, cmd_args):
    try:
        int_arg = int(cmd_args[name])
    except:
        try:
            int_arg = cmd_args["default_int"]
        except:
            int_arg = None
    return int_arg
    
def _get_file_extension_from_url(url):
    tokens = os.path.splitext(os.path.basename(urllib.parse.urlsplit(url).path))
    if len(tokens) > 1:
        return tokens[1]
    return ""
    
async def _download_to_tmp(url):
    global ACCEPTED_ATTACHMENTS
    url_ext = _get_file_extension_from_url(url).lower()
    if not (url_ext in ACCEPTED_ATTACHMENTS):
        return ""
    
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
    
def _run_string(run_string):   # run shell command asynchronously to keep discord message pumps happy and allow cancellation
    global ROOT_PATH
    os.chdir(ROOT_PATH)
    print(run_string)
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
    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)
    
    output_attachments = []
    try:
        user = ctx.message.author.name
        author = user
        
        if "-user" in cmd_args:
            user = cmd_args["-user"]
        
        select_num = _get_int_arg("-num", cmd_args)
        if select_num == None: select_num = 1
            
        output_attachments = CMD_QUEUE.get_last_output(user=user)
        if len(output_attachments) == 0:
            raise Exception("No output images to select")
            
        output_attachments = [output_attachments[select_num-1]]
            
        _reversed = reversed(CMD_QUEUE.cmd_list) # look for the most recently completed command by the requesting user with an attached image and replace it with the selected one
        for cmd in _reversed:
            if cmd.init_user == author:
                if cmd.status == 2: # completed successfully
                    cmd.in_attachments = output_attachments
                    break
        
    except Exception as e:
        await ctx.send("Sorry @" + ctx.message.author.name + ", " + str(e))
    
    return output_attachments
    
@client.event
async def on_ready():

    global BOT_ACTIVITY
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=BOT_ACTIVITY))
    
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
    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)
    
    global BOT_ADMIN_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME]): return
        
    CMD_QUEUE.restart_now = ctx
    
    if "-force" in cmd_args:
        for cmd in CMD_QUEUE.cmd_list: 
            if cmd.status in [0, 1]: # cancel everything running or waiting
                cmd.status = 3 # cancelled
        
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

    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)
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
    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)
    
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
        if cmd.status in [0, 1]:
            if user == "*":
                cmd.status = 3 # cancelled
            else:
                if cmd.init_user.strip().lower() == user.strip().lower():
                    cmd.status = 3 # cancelled
        
    return

@client.command()
@commands.is_owner()
async def clean(ctx): # clean all temp folders (only used by the bot owner)
    
    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)
    
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
    
    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)    
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    
    if "-all" in cmd_args:
        for cmd in CMD_QUEUE.cmd_list:
            if cmd.init_user == ctx.message.author.name:
                if cmd.status in [0, 1]: # queued or in-progress
                    cmd.status = 3 # cancelled

        await ctx.send("Okay @" + ctx.message.author.name + ", cancelling all your queued commands...")
        return
        
    else:
    
        num_to_cancel = _get_int_arg("-x", cmd_args)
        if num_to_cancel == None: num_to_cancel = 1
        
        num_cancelled = 0
        for i in range(num_to_cancel):

            cmd = CMD_QUEUE.get_last_command(user=ctx.message.author.name, status_list=[0,1])   # queued or in-progress
            if cmd:
                cmd.status = 3 # cancelled
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
    global PARAM_LIST
    command, cmd_args = _parse_args(ctx.message.content, PARAM_LIST)
    
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
       
@tasks.loop(seconds = QUEUE_POLL_INTERVAL) # repeat at polling interval
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
    
    cmd.status = 1 # running
    
    # don't think I'll actually this, it reduces performance so disabling for now
    #command_server_status = await CMD_QUEUE.get_command_server_status()
    #if not command_server_status:           # verify the command server is running and responsive
    #    CMD_QUEUE.restart_command_server()  # if it isn't, forcibly kill the whole process and restart it

    auto_seed = str(np.random.randint(AUTO_SEED_RANGE[0], AUTO_SEED_RANGE[1]))  # create an auto seed
    default_params = DEFAULT_CMD_PARAMS.copy()
    default_params["-seed"] = auto_seed
    cmd.cmd_args = _merge_dicts(default_params, cmd.cmd_args) # add default params
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CMD_QUEUE.cmd_server_url, json=cmd.__getstate__()) as response:
                json_data = await response.json()

    except Exception as e:
        json_data = None
        cmd.status = -1 # error status
        cmd.error_txt = "Error sending command to command server - " + str(e)
        
    if cmd.status == 3: # cancelled status
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
            cmd.status = -1 # error status
            cmd.error_txt = "Error parsing results from command server - "  + str(e)
        
    next_cmd = CMD_QUEUE.get_next_pending()
    
    if cmd.status == 2: # completed successfully
    
        # update the per user total run-time cache
        if cmd.init_user in CMD_QUEUE.users_elapsed_time.keys():
            CMD_QUEUE.users_elapsed_time[cmd.init_user] += cmd.elapsed_time
        else:
            CMD_QUEUE.users_elapsed_time[cmd.init_user] = cmd.elapsed_time

        # send requesting user their output attachments
        try:
            files = []
            if cmd.out_preview_image != "": # use summary / preview image if available and send only 1 attachment
                files.append(discord.File(cmd.out_preview_image))
            else:
                for out_file in cmd.out_attachments:
                    files.append(discord.File(out_file))
                    
            msg = "Finished " + cmd.get_summary(no_init_time=True)
            if next_cmd == None: msg += " - Queue is empty!"
            if cmd.error_txt != "": msg += "\n" + cmd.error_txt
            
            await cmd.ctx.send(files=files, content=msg)
        except Exception as e:
            cmd.status = -1 # error
            cmd.error_txt = "Error sending output image - " + str(e)
            
    if cmd.status == -1: # error
        
        try:
            print("Error processing command - " + cmd.error_txt)
            msg = "Sorry @" + cmd.init_user + ", " + cmd.error_txt + " - " + cmd.get_summary()
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

    global BOT_COMMAND_PREFIX
    global BOT_COMMAND_LIST
    global BOT_PARAM_LIST
    
    if message.author.bot: return
    if message.content.startswith(BOT_COMMAND_PREFIX) != True: return
    
    command, cmd_args = _parse_args(message.content, PARAM_LIST)
    if command.lower().replace(BOT_COMMAND_PREFIX, "") not in BOT_COMMAND_LIST: return
    
    await client.process_commands(message)

@client.event
async def on_disconnect():
    await asyncio.sleep(10)


if __name__ == "__main__":

    # this bot requires both message and message content intents (message content is a privileged intent)
    intents = discord.Intents().default()
    intents.messages = True
    intents.dm_messages = True
    intents.message_content = True

    client = commands.Bot(command_prefix=BOT_COMMAND_PREFIX, intents = intents)
    client.remove_command('help') # required to make the custom help command work

    CMD_QUEUE = CommandQueue(queue_mode=QUEUE_MODE)
    client.run(BOT_TOKEN, reconnect=True)