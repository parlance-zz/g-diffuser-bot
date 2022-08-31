"""

 G-DiffuserBot for Discord (https://https://github.com/parlance-zz/g-diffuser-bot/)
 
 todo:
 1. add command list trimming by adding a replacement dummy completed command with total elapsed time and in out image props (_compactify)
 2. change _auto_clean to not delete files referenced by any input or output image in the command queue, unless -force
 3. cleanup param reformatting since long commands are no longer supported
 4. cleanup _process_commands to merge i2i and t2i because there is a lot of redundant code now
 5. fixup path concat to use os.path, low priority because no user data is used
"""


# --- bot params ----------------------------------------------------------------------------

BOT_TOKEN = "YOUR_BOT_TOKEN_GOES_HERE"
BOT_ADMIN_ROLE_NAME = "AI Overlords"   # use your group names here accordingly, check the help text for command permissions
BOT_USERS_ROLE_NAME = "AI Underlings"

BOT_COMMAND_PREFIX = "!"
BOT_ACTIVITY = "The way of the future..."

# default paths - please adjust accordingly
SD_ROOT_PATH = "C:/stable-diffusion/stable-diffusion-main"                            # make this your stable-diffusion-main root path
TMP_ROOT_PATH = SD_ROOT_PATH + "/scripts/tmp"                                         # make sure this is a valid path (it will be created if it does not exist)
ESRGAN_PATH = SD_ROOT_PATH + "/scripts/esrgan/realesrgan-ncnn-vulkan.exe"             # optional, but required for !enhance and !txt2imghd
BOT_PATH = "scripts/g-diffuser-bot"
BOT_STATE_DATA_FILE = BOT_PATH + "/g-diffuser-bot.pickle"  # can be disabled, used for persisting command queue, top user list and input image paths

# default params for commands, these override any user supplied params (except seed)
TXT2IMG_ADD_PARAMS = { "--n_samples": "2" }
IMG2IMG_ADD_PARAMS = {}
ESRGAN_ADD_PARAMS = { "-n": "realesrgan-x4plus" }
AUTO_SEED_RANGE = (1,999999)
MAX_STEPS_LIMIT = 120
MAX_REPEAT_LIMIT = 100             # max number of repititions that can be used with the -x param

MAX_QUEUE_LENGTH = 1000            # beyond this limit additional commands will be rejected
QUEUE_MODE = 0                     # 0 for round-robin, 1 for first come first serve
MAX_QUEUE_PRINT_ITEMS = 4          # max number of items to show for !queue command (up to discord message length limit)

SHORTHAND_LIST = { "-str": "--strength", "-scale": "--scale", "-seed": "--seed", "-plms": "--plms", "-steps": "--ddim_steps", "-n": "--n_samples", "-x": "-x", "-m": "--ckpt" } # shortcut map for command params

TMP_CLEAN_PATHS = [
    SD_ROOT_PATH + "/outputs/img2img-samples/*.png",
    SD_ROOT_PATH + "/outputs/txt2img-samples/*.png",
    SD_ROOT_PATH + "/outputs/txt2imghd-samples/*.png",
    SD_ROOT_PATH + "/outputs/img2img-samples/samples/*.png",
    SD_ROOT_PATH + "/outputs/txt2img-samples/samples/*.png",
    SD_ROOT_PATH + "/outputs/txt2imghd-samples/samples/*.png",
    SD_ROOT_PATH + "/scripts/tmp/*.png",
    SD_ROOT_PATH + "/scripts/tmp/*.pickle",
]

GRID_OUTPUT_PADDING = 2

# -------------------------------------------------------------------------------------------

# these strings must be 2000 characters or less

ABOUT_TXT = """This is a simple discord bot for stable-diffusion and provides access to the most common commands as well as a few others.

Commands can be used in any channel the bot is in, provided you have the appropriate server role. For a list of commands, use !help

Please use discretion in your prompts as the safety filter has been disabled. Repeated violations will result in banning.
If you do happen to generate anything questionable please delete the message yourself or contact a mod ASAP. The watermarking feature has been left enabled to minimize potential harm.

For more information on the G-DiffuserBot please see https://github.com/parlance-zz/g-diffuser-bot
"""

HELP_TXT1 = """
User Commands:
  !t2i : Generates an image with a prompt [-seed num] [-scale num] [-steps num] [-plms] [-m model] [-x num]
  !t2ihd : As above but no -plms support, uses txt2imghd to generate 1 sample at 4x size
  !i2i : Generates an image with a prompt and input image [-seed num] [-str num] [-scale num] [-steps num] [-m model] [-x num] 
  !enhance : Uses esrgan to upscale the input image image by 4x
  !queue : Shows running / waiting commands in the queue [-mine]
  !cancel : Cancels your last command (can be used while running) [-all]
  !top : Shows the top users' total running time
  !select : Selects an image by number from your last result and make it your input image (left to right, top to bottom) (skips the queue)
  !show_input : Shows your current input image (skips the queue)
 
Admin Commands:
  !shutdown : Cancel all pending / running commands and shutdown the bot (can only be used by bot owner)
  !clean : Delete temporary files in SD folders, -force will delete temporary files that may still be referenced (can only be used by bot owner) [-force]
  !restart : Restart the bot after the command queue is idle
  !clear [user]: Cancel all or only a specific user's pending / running commands
"""
HELP_TXT2=""" 
Parameter Notes:
  -seed : Any whole number (default random)
  -scale : Can be any positive real number (default 6). Controls the unconditional guidance scale. Good values are between 3-20.
  -str : Number between 0 and 1, (default 0.4). Controls how much to change the input image. 
  -plms : Use the plms instead of ddim sampler to generate your output.
  -steps: Any whole number from 10 to 200 (default 50). Controls how many times to recursively change the input image.
  -x: Repeat the given command some number of times. The number of possible repeats may be limited.
  -m: Choose the model to use

Input images:
  Commands that require an input image will use the image you attach to your message. If you do not attach an image it will attempt to use the last image you attached.
  Input images will be cropped to 1:1 aspect and resized to 512x512.
  The select command can be used to turn your last command's output image into your next input image, please see !select above.

Examples:
  To see examples of valid commands use !examples
"""

EXAMPLES_TXT = """
Example commands:
!t2i an astronaut riding a horse on the moon
!t2i painting of an island by lisa frank -plms -seed 10
!t2ihd baroque painting of a mystical island treehouse on the ocean, chrono trigger, snes style, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate -plms
!t2i my little pony in space marine armor from warhammer 40k, trending on artstation, intricate detail, 3d render, gritty, dark colors, cinematic lighting, cosmic background with colorful constellations -scale 10 -seed 174468 -steps 50
"""

# -------------------------------------------------------------------------------------------



# discord and psutil are the only modules you need outside of the pre-built ldm conda environment

try: # install discord module if we haven't already
    import discord
except ImportError:
    from pip._internal import main as pip
    pip(['install', 'discord'])
    import discord

from discord.ext import commands
from discord.ext import tasks

# psutil is used to reliably force-close commands that have been cancelled while running
try: # install psutil module if we haven't already
    import psutil
except ImportError:
    from pip._internal import main as pip
    pip(['install', 'psutil'])
    import psutil

# these dependencies are part of the conda ldm environment for the public release of stable-diffusion 1.4
import subprocess
import glob
import os
import sys
import random
import time
import asyncio
import requests
import uuid
import shutil
from PIL import Image
import numpy as np
import shlex
import datetime
import string
from pathlib import Path
import copy

# pickle is only used to save the command queue / remember input images, output images and user total run-time between restarts
# if you set BOT_STATE_DATA_FILE = "" in the global options below then you can safely remove this import
import pickle

try: # try to make sure temp folder exists
    os.makedirs(TMP_ROOT_PATH)
except Exception as e:
    print("Error creating temp path: '" + TMP_ROOT_PATH + "' - " + str(e))

USERS_ELAPSED_TIME = {}
RESTART_NOW = None
SCRIPT_FILE_NAME = os.path.basename(__file__)
print(SCRIPT_FILE_NAME)

#intents = discord.Intents(messages=True, members=True, presences=True)
intents = discord.Intents.all() # if you need to be more strict with your discord intents please adjust accordingly

client = commands.Bot(command_prefix=BOT_COMMAND_PREFIX, intents = intents)
client.remove_command('help') # required to make the custom help command work
game = discord.Game(name=BOT_ACTIVITY)

class Command:

    def __init__(self, ctx=None):
    
        self.ctx = ctx  # discord context is attached to command to reply appropriately
        self.init_time = datetime.datetime.now()
        self.init_user = ctx.message.author.name
        self.message = ctx.message.content 
        self.command = ctx.message.content.split()[0].strip().lower()
        self.cmd_args = _parse_args(ctx.message.content)
        self.run_path = ""             # final command line executed
        self.in_image = ""             # local path to input image
        self.out_image = ""            # local path to output image
        self.out_image_dims = (0, 0)   # size of the individual images in the output 
        self.out_image_layout = (1, 1) # grid layout of image, or (1,1) for a single image
        
        self.start_time = datetime.datetime.max
        self.elapsed_time = datetime.timedelta(seconds=0)
        self.status = 0 # 0 for waiting in queue, 1 for running, 2 for run successfully, 3 for cancelling, -1 for error
        self.error_txt = ""
    
    def __getstate__(self):      # remove discord context for pickling
        attributes = self.__dict__.copy()
        if "ctx" in attributes.keys():
            del attributes["ctx"]
        return attributes
        
    def get_summary(self):
        
        summary = ("[" + self.init_time.strftime("%I:%M:%S %p")).replace("[0", "[") + "]  "
        summary += "@" + self.init_user + "  " + self.message
        
        if "--seed" in self.cmd_args:
            summary += " [seed:" + self.cmd_args["--seed"] + "]"
            
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

    def __init__(self, data_file=""):
        
        global SD_ROOT_PATH
        global USERS_ELAPSED_TIME
        global TMP_ROOT_PATH
        
        self.cmd_list = []
        self.data_file_path = ""

        if data_file != "": # load existing data if given a data file path
                        
            self.data_file_path = data_file
            
            try:
                os.chdir(SD_ROOT_PATH)
                
                file_size = os.stat(data_file).st_size
                if file_size == 0:
                    list_of_files = glob.glob(TMP_ROOT_PATH + "/*.pickle")
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print("'" + data_file + "' is empty, loading '" + latest_file + "' instead...")
                    self.data_file_path = latest_file
                    data_file = latest_file
                    
                file = open(data_file, "rb")
                self.cmd_list = pickle.load(file)
                file.close()
                
                for cmd in self.cmd_list:
                    if cmd.init_user in USERS_ELAPSED_TIME.keys():
                        USERS_ELAPSED_TIME[cmd.init_user] += cmd.elapsed_time
                    else:
                        USERS_ELAPSED_TIME[cmd.init_user] = cmd.elapsed_time
            
                print("Loaded " + data_file + "...")
                
            except:
                print("Error loading " + data_file)
        
    def save(self):
        
        global SD_ROOT_PATH
        
        if self.data_file_path != "": # save command queue data if we have a data file path
            try:
                os.chdir(SD_ROOT_PATH)
                
                file = Path(self.data_file_path)
                file.touch(exist_ok=True)
                
                old_file_size = os.stat(self.data_file_path).st_size
                if old_file_size > 0:
                    backup_path = _get_tmp_path(".pickle")
                    shutil.copyfile(self.data_file_path, backup_path) # copy to make a backup file (if not empty)
                    print("Backed up data file to " + backup_path + "...")
                
                file = open(self.data_file_path, "wb")
                
                cmd_list_copy = []
                
                for cmd in self.cmd_list:
                    if cmd.status == 2: # only save completed commands (because we can't save contexts for waiting commands)
                        cmd_list_copy.append(cmd)
                    
                pickle.dump(cmd_list_copy, file)
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
    
        global QUEUE_MODE
        
        pending_list = []
        
        if QUEUE_MODE == 0: # round-robin
        
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
                        break
        
        if num_to_return == 1: # rather than return an empty list this command returns None or the actual command when num_to_return == 1
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
                if cmd.in_image != "":
                    return cmd.in_image
                    
        return ""
        
    def get_last_output(self, user=None, num=1, select_mode=0): # get user's last output image, cropped if grid
        
        global GRID_OUTPUT_PADDING
        
        _reversed = reversed(self.cmd_list)
        for cmd in _reversed:
            if (cmd.init_user == user) or (user == None):
                if (cmd.status == 2) and (cmd.out_image != ""):  # completed successfully
                
                    if cmd.out_image_layout == (1, 1): # not a grid
                        return cmd.out_image
                
                    try:
                    
                        total_num_outputs = cmd.out_image_layout[0] * cmd.out_image_layout[1]
                        if num > total_num_outputs:
                            num = total_num_outputs
                        
                        if select_mode == 0: # reasonable people would only ever use 0, the default numbering scheme
                            tile_loc = ((num-1) % cmd.out_image_layout[0], (num-1) // cmd.out_image_layout[0])
                        else:
                            tile_loc = ((num-1) // cmd.out_image_layout[0], (num-1) % cmd.out_image_layout[0])
                            
                        crop_loc = (tile_loc[0] * (cmd.out_image_dims[0] + GRID_OUTPUT_PADDING) + GRID_OUTPUT_PADDING, tile_loc[1] * (cmd.out_image_dims[1] + GRID_OUTPUT_PADDING) + GRID_OUTPUT_PADDING)
                    
                        # crop out from the grid
                        img = Image.open(cmd.out_image)
                        crop_area = (crop_loc[0], crop_loc[1], crop_loc[0] + cmd.out_image_dims[0], crop_loc[1] + cmd.out_image_dims[1])
                        img = img.crop(crop_area)
                        tmp_path = _get_tmp_path(".png")
                        img.save(tmp_path)
                        img.close()
                    
                        return tmp_path
                        
                    except Exception as e:
                        print("Error selecting img " + str(num) + " - " + str(e))
                        return ""
        
        return ""        
        
    async def add_new(self, ctx): # add a new command to the queue
        
        global MAX_REPEAT_LIMIT
        global MAX_QUEUE_LENGTH
        
        rejected = (RESTART_NOW != None) or (self.get_queue_length() >= MAX_QUEUE_LENGTH)
        x_val = ""
        
        if rejected:
            try:
                if RESTART_NOW != None:
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
            if num_attachments > 0:
                url = ctx.message.attachments[0].url.lower()
                cmd.in_image = await _download_to_tmp(url)
        except Exception as e:
            print("Error downloading command attachment - " + str(e))
        
        self.cmd_list.append(cmd)
        
        try:
            await ctx.send("Okay @" + cmd.init_user + ", gimme a sec...")
        except Exception as e:
            print("Error sending acknowledgement - " + str(e))
        
        if x_val != "": # repeat command           
                    
            try:
                repeat_x = int(x_val) - 1
                max_repeat_limit = MAX_REPEAT_LIMIT
                if (MAX_QUEUE_LENGTH - self.get_queue_length()) < max_repeat_limit:
                    max_repeat_limit = MAX_QUEUE_LENGTH - self.get_queue_length()
                    
                if repeat_x >= max_repeat_limit:
                    repeat_x = max_repeat_limit - 1
                
                for x in range(repeat_x):
                    cmd_copy = Command(ctx)
                    del cmd_copy.cmd_args["-x"]
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
        
def _shlex_split(value):    #  did this to make things easier on windows
    lex = shlex.shlex(value)
    lex.quotes = '"'
    lex.whitespace_split = True
    lex.commenters = ''
    return list(lex)
    
def _parse_args(msg):

    global SHORTHAND_LIST
    
    tokens = _shlex_split(msg)

    args = {}
    if len(tokens) < 2: return args
    
    last_key = None
    for i in range(1, len(tokens)):
        if (tokens[i] in SHORTHAND_LIST.values()) or (tokens[i] in SHORTHAND_LIST.keys()):
            args[tokens[i]] = True
            last_key = tokens[i]
        else:
            if last_key:
                args[last_key] = tokens[i]
                last_key = None
            else:
                args[tokens[i]] = True
    
    return args

def _join_args(cmd, args):
    
    arg_list = []

    for arg in args:
        if args[arg] == True:
            arg_list.append(arg)
        else:
            val = args[arg]
            if " " in val.strip():
                val = '"' + val.strip() + '"'
            arg_list.append(arg)
            arg_list.append(val)
    
    # todo: i'd like to use shlex join but it insists on using single quotes for no reason
    #arg_list = shlex.join(arg_list)
    if len(arg_list) > 0:
        arg_list.insert(0, cmd)
        cmd = " ".join(arg_list)
        #cmd = cmd + " " + arg_list
    #return cmd.replace("'", '"')
    return cmd
    
def _short_cmd_reformat(args):
    
    global SHORTHAND_LIST
    
    short_dict = SHORTHAND_LIST
    new_args = {}
    
    for key in short_dict:
        if key in args:
            new_args[short_dict[key]] = args[key]
            del args[key]
    
    prompt = " ".join(args.keys())
    new_args["--prompt"] = prompt.replace('"', "")
    
    return new_args
    
async def _download_to_tmp(url):

    url = url.lower()
    print("Downloading '" + url + "'...")
    request = requests.get(url, stream = True)
    
    url_ext = url[url.rfind("."):]
    tmp_file_path = _get_tmp_path(url_ext)
    
    with open(tmp_file_path, "wb") as out_file:
        shutil.copyfileobj(request.raw, out_file)
        
    return tmp_file_path
    
def _get_tmp_path(file_extension):
    global TMP_ROOT_PATH
    return TMP_ROOT_PATH + "/" + str(uuid.uuid4()) + file_extension
    
def _get_image_dims(img_path):
    img = Image.open(img_path)
    size = img.size
    img.close()
    return size
    
def _regularize_image(in_path, out_path, dims): # format an image for 1:1 aspect for img2img

    img = Image.open(in_path)
    
    width, height = img.size
    ldim = np.minimum(width, height)
    area = (width//2 - ldim//2, height//2 - ldim//2, width//2 + ldim//2, height//2 + ldim//2)
    
    img = img.crop(area)
    img = img.resize(dims)
    img.save(out_path)
    img.close()
    
    return

def _merge_dicts(_d1, d2):
    d1 = _d1.copy()
    for x in d2:
        d1[x] = d2[x]
    return d1

def _restart_program():
    
    global SD_ROOT_PATH
    global SCRIPT_FILE_NAME
    global BOT_PATH
    
    print("Restarting...")
    os.chdir(SD_ROOT_PATH)
    run_string = 'python "' + BOT_PATH + '/' + SCRIPT_FILE_NAME + '"'
    print(run_string)
    subprocess.Popen(run_string)
    exit()
    
def _auto_clean():  # delete images and pickles from temporary and output paths

    global TMP_CLEAN_PATHS

    for path in TMP_CLEAN_PATHS:
        try:
            file_list = glob.glob(path)
            for file_path in file_list:
                try:
                    print("Removing '" + file_path + "'...")
                    os.remove(file_path)
                except:
                    unable_to_remove = True
            
            print("Cleaned " + path)
        except Exception as e:
            print("Error cleaning - " + path + " - " + str(e))
            
    return
    
def _check_server_roles(ctx, role_name_list):
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
    
def _p_kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
    return
    
async def _run_cmd(run_string, cmd):   # run shell command asynchronously to keep discord message pumps happy and allow cancellation

    global SD_ROOT_PATH
    
    os.chdir(SD_ROOT_PATH)
    print(run_string)
    process = subprocess.Popen(run_string, shell=True)

    while process.poll() == None:
    
        if cmd.status == 3: # cancelled
            _p_kill(process.pid)
            print("Killing process - " + str(process.pid))
            return -1
                
        await asyncio.sleep(1)
        
    process.wait()
    return process.returncode
    
async def _top(ctx):    # replies to a message with a sorted list of all users and their run-time

    global USERS_ELAPSED_TIME
    
    msg = "Okay @" + ctx.message.author.name + ", here's the top users... \n"
    i = 0
    for user in sorted(USERS_ELAPSED_TIME, reverse=True, key=USERS_ELAPSED_TIME.get):
        i += 1
        msg += str(i) + ": @" + user + " <" + str(datetime.timedelta(seconds=USERS_ELAPSED_TIME[user].seconds)) + "s>\n"
    if i == 0:
        msg = "No users yet!"
        
    await ctx.send("@" + ctx.message.author.name + " : " + msg)
    
async def _select(ctx, select_mode=0): # crop an image from the user's last output image grid
    
    global CMD_QUEUE
    
    try:
        select_num = 1
        tokens = ctx.message.content.split()
        user = ctx.message.author.name
        
        if len(tokens) > 1:
            try:
                select_num = int(tokens[1].strip())
            except Exception as e:
                raise Exception("Invalid argument in select - " + str(e))
            
        input_path = CMD_QUEUE.get_last_output(user=user, num=select_num, select_mode=select_mode)
        if input_path == "":
            raise Exception("No output images to select")
        
        found_cmd = False
        _reversed = reversed(CMD_QUEUE.cmd_list) 
        for cmd in _reversed:
            if cmd.init_user == user:
                if cmd.status == 2: # completed successfully
                    cmd.in_image = input_path
                    found_cmd = True
                    break
        
        if not found_cmd:
            raise Exception("No output images to select")
        
    except Exception as e:
        await ctx.send("Sorry @" + ctx.message.author.name + ", " + str(e))
        return    
    
@client.event
async def on_ready():

    global game
    
    await client.change_presence(activity=game)
    _process_commands_loop.start()
    print("The bot is ready!")
    return
    
@client.command()
@commands.is_owner()
async def shutdown(ctx): # shutdown the bot (only used by the bot owner)
    await ctx.send("Bye")
    exit()
    
@client.command()
async def restart(ctx): # restart the bot when the queue is empty (available to admins)

    global RESTART_NOW
    
    global BOT_ADMIN_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME]): return
        
    RESTART_NOW = ctx
    try:
        await ctx.send("Okay @" + ctx.message.author.name + ", restarting when queue is empty... ")
    except Exception as e:
        print("Error sending restart acknowledgement - " + str(e))
        
    return
    
@client.command()
async def clear(ctx): # clear the command queue completely (available to admins)
    
    global CMD_QUEUE
    
    global BOT_ADMIN_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME]): return
        
    tokens = ctx.message.content.split()
    user = ""
    if len(tokens) > 1:
        user = tokens[1].strip().lower()
    
    if user == "":
        try:
            await ctx.send("Okay @" + ctx.message.author.name + ", clearing the queue... ")
        except:
            print("Error sending acknowledgement")
    else:
        try:
            await ctx.send("Okay @" + ctx.message.author.name + ", clearing @" + tokens[1].strip() + " from the queue... ")
        except:
            print("Error sending acknowledgement")
            
    for cmd in CMD_QUEUE.cmd_list:
        if cmd.status in [0, 1]:
            if user == "":
                cmd.status = 3 # cancelled
            else:
                if cmd.init_user.strip().lower() == user:
                    cmd.status = 3 # cancelled
        
    return

@client.command()
@commands.is_owner()
async def clean(ctx): # clean all temp folders (only used by the bot owner)
    
    await ctx.send("Okay @" + ctx.message.author.name + ", cleaning temp files... ")
    _auto_clean()
        
    return
    
@client.command()
async def cancel(ctx): # stops the requesting user's last queued command, or all of them

    global CMD_QUEUE
    
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    
    tokens = ctx.message.content.split()
    param = ""
    if len(tokens) > 1:
        param = tokens[1].strip().lower()
    
    if param == "-all":
    
        for cmd in CMD_QUEUE.cmd_list:
            if cmd.init_user == ctx.message.author.name:
                if cmd.status in [0, 1]: # queued or in-progress
                    cmd.status = 3 # cancelled
                    
        await ctx.send("Okay @" + ctx.message.author.name + ", cancelling all queued commands...")
        
    else:
    
        cmd = CMD_QUEUE.get_last_command(user=ctx.message.author.name, status_list=[0,1])   # queued or in-progress
        if cmd:
            cmd.status = 3 # cancelled
            await ctx.send("Okay @" + ctx.message.author.name + ", " + cmd.get_summary())
        else:
            await ctx.send("Sorry @" + ctx.message.author.name + ", no running or waiting commands to cancel...")
            
    return
    
@client.command()
async def show_input(ctx): # attaches the requesting user's input image in response

    global CMD_QUEUE
    
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
        
    input_path = CMD_QUEUE.get_last_attached(user=ctx.message.author.name)
    if input_path != "":
        try:
            file = discord.File(input_path)
            msg = "@" + ctx.message.author.name + ", this is your current input image"
            await ctx.send(file=file, content=msg)
        except Exception as e:
            print("Error sending show user input image - " + str(e))
            try:
                await ctx.send("Sorry @" + ctx.message.author.name + ", your last attachment is too old...")
            except:
                print("")
        
    else:
        try:
            await ctx.send("Sorry @" + ctx.message.author.name + ", no input image to show")
        except Exception as e:
            print("Error sending show user input image rejection - " + str(e))
    return
    
@client.command()
async def hello(ctx):
    await ctx.send("Hi")
    
@client.command() # show the next part of the pending command list
async def queue(ctx):

    global CMD_QUEUE
    
    tokens = ctx.message.content.split()
    param = ""
    if len(tokens) > 1:
        param = tokens[1].strip().lower()
    
    if param == "-mine":
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
    await _select(ctx,select_mode=0)
    await show_input(ctx)
    return
    
@client.command() # an alternate numbering scheme in case you don't know how to count properly
async def gselect(ctx):
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return
    await _select(ctx,select_mode=1)
    await show_input(ctx)
    return
    
@tasks.loop(seconds = 1) # repeat after every second
async def _process_commands_loop():

    global TXT2IMG_ADD_PARAMS
    global IMG2IMG_ADD_PARAMS
    global AUTO_SEED_RANGE
    global SD_ROOT_PATH
    global ESRGAN_PATH
    global ESRGAN_ADD_PARAMS
    global CMD_QUEUE
    global RESTART_NOW
    global MAX_STEPS_LIMIT
    global USERS_ELAPSED_TIME
    global BOT_STATE_DATA_FILE
    
    if CMD_QUEUE.get_running(): return # don't re-enter if something is already running
    CMD_QUEUE.clear(status_list = [3]) # clear cancelled
    cmd = CMD_QUEUE.get_next_pending()
    
    if cmd == None:
        if RESTART_NOW:
            print("Restarting now...")
            try:
                await RESTART_NOW.send("Restarting now...")
            except:
                print("Error sending restart message")
            try:
                await asyncio.sleep(2)
                _restart_program()
            except:
                exit()
                
        return # nothing to do
    
    auto_seed = str(np.random.randint(AUTO_SEED_RANGE[0], AUTO_SEED_RANGE[1]))
    
    if cmd.command in ["!t2i", "!t2ihd"]:
    
        try:

            cmd.status = 1 # in-progress
            cmd.start_time = datetime.datetime.now()
            
            hd = ("hd" in cmd.command)       
            
            if not hd:
                cmd.run_string = "python scripts/txt2img.py"
            else:
                cmd.run_string = "python scripts/txt2imghd.py"
            
            if not "--prompt" in cmd.cmd_args:
                cmd.cmd_args = _short_cmd_reformat(cmd.cmd_args)
            if not hd: cmd.cmd_args = _merge_dicts(cmd.cmd_args, TXT2IMG_ADD_PARAMS)
            if not "--seed" in cmd.cmd_args:
                cmd.cmd_args["--seed"] = auto_seed
            if "--ckpt" in cmd.cmd_args:
                cmd.cmd_args["--ckpt"] = SD_ROOT_PATH + "/models/ldm/stable-diffusion-v1/" + cmd.cmd_args["--ckpt"] + ".ckpt"
            if "--ddim_steps" in cmd.cmd_args:
                try:
                    steps = int(cmd.cmd_args["--ddim_steps"])
                    if steps > MAX_STEPS_LIMIT:
                        steps = MAX_STEPS_LIMIT
                        cmd.cmd_args["--ddim_steps"] = str(steps)
                except:
                    steps = 0
                    
            if hd and ("--ddim_steps" in cmd.cmd_args):
                cmd.cmd_args["--steps"] = cmd.cmd_args["--ddim_steps"]
                del cmd.cmd_args["ddim_steps"]
            if hd and ("--plms" in cmd.cmd_args):
                del cmd.cmd_args["--plms"]
                
            cmd.run_string = _join_args(cmd.run_string, cmd.cmd_args)
            return_code = await _run_cmd(cmd.run_string, cmd)
            
            if cmd.status != 3: # cancelled
                if return_code != 0:
                    cmd.err_txt = "Error returned by command"
                    cmd.status = -1  # error
                else:
                    if not hd: # success
                        list_of_files = glob.glob('outputs/txt2img-samples/*.png')
                        cmd.out_image_dims = (512, 512)
                        cmd.out_image_layout = (2, 2)
                    else:
                        list_of_files = glob.glob('outputs/txt2imghd-samples/samples/*.png')
                        cmd.out_image_dims = (512 * 4, 512 * 4)
                        cmd.out_image_layout = (1, 1)
                        
                    latest_file = max(list_of_files, key=os.path.getctime)
                    print(latest_file)
                    cmd.out_image = latest_file
                    cmd.status = 2 # success
                    
        except Exception as e:
            cmd.status = -1 # error
            cmd.err_txt = str(e)
    
    elif cmd.command in ["!i2i"]:
    
        try:
            cmd.status = 1 # in-progress
            cmd.start_time = datetime.datetime.now()
            
            input_path = cmd.in_image
            if input_path == "": input_path = CMD_QUEUE.get_last_attached(user=cmd.init_user)
            if input_path == "":
                cmd.err_txt = "No attached image"
                cmd.status = -1 # error
            else:
                # crop to largest square then resize to native 512x512
                reg_path = _get_tmp_path(".png")
                _regularize_image(input_path, reg_path, (512, 512))
                
                if not "--prompt" in cmd.cmd_args:
                    cmd.cmd_args = _short_cmd_reformat(cmd.cmd_args)
                cmd.cmd_args = _merge_dicts(cmd.cmd_args, IMG2IMG_ADD_PARAMS)
                if not "--seed" in cmd.cmd_args:
                    cmd.cmd_args["--seed"] = auto_seed                    
                cmd.cmd_args["--init-img"] = '"' + reg_path + '"'
                if "--ckpt" in cmd.cmd_args:
                    cmd.cmd_args["--ckpt"] = SD_ROOT_PATH + "/models/ldm/stable-diffusion-v1/" + cmd.cmd_args["--ckpt"] + ".ckpt"
                if "--ddim_steps" in cmd.cmd_args:
                    try:
                        steps = int(cmd.cmd_args["--ddim_steps"])
                        if steps > MAX_STEPS_LIMIT:
                            steps = MAX_STEPS_LIMIT
                            cmd.cmd_args["--ddim_steps"] = str(steps)
                    except:
                        steps = 0
                
                if "--plms" in cmd.cmd_args:
                    del cmd.cmd_args["--plms"]
                
                cmd.run_string = _join_args("python scripts/img2img.py", cmd.cmd_args)
                return_code = await _run_cmd(cmd.run_string, cmd)
                
                if cmd.status != 3: # cancelled
                    if return_code != 0:
                        cmd.err_txt = "Error returned by command"
                        cmd.status = -1  # error
                    else:
                        list_of_files = glob.glob('outputs/img2img-samples/*.png')
                        latest_file = max(list_of_files, key=os.path.getctime)
                        print(latest_file)
                        cmd.out_image_dims = (512, 512)
                        cmd.out_image_layout = (2, 1)
                        cmd.out_image = latest_file
                        cmd.status = 2 # success
                
        except Exception as e:
            cmd.status = -1 # error
            cmd.err_txt = str(e)
        
    elif cmd.command in ["!enhance"]:
    
        try:
            cmd.status = 1 # in-progress
            cmd.start_time = datetime.datetime.now()
            
            input_path = cmd.in_image
            if input_path == "": input_path = CMD_QUEUE.get_last_attached(user=cmd.init_user)
            if input_path == "":
                cmd.err_txt = "No attached image"
                cmd.status = -1 # error
            else:
            
                if not input_path.endswith(".png"):
                    img = Image.open(input_path)
                    tmp_file_path = _get_tmp_path(".png")
                    img.save(tmp_file_path)
                    img.close()
                    input_path = tmp_file_path

                enhanced_file_path = _get_tmp_path(".png")
                cmd.cmd_args = _merge_dicts(cmd.cmd_args, ESRGAN_ADD_PARAMS)
                cmd.cmd_args["-i"] = '"' + input_path + '"'
                cmd.cmd_args["-o"] = '"' + enhanced_file_path + '"'
                
                cmd.run_string = _join_args(ESRGAN_PATH, cmd.cmd_args)
                return_code = await _run_cmd(cmd.run_string, cmd)
                
                if cmd.status != 3: # cancelled
                    if return_code != 0:
                        cmd.err_txt = "Error returned by command"
                        cmd.status = -1  # error
                    else:
                        print(enhanced_file_path)
                        cmd.out_image = enhanced_file_path
                        cmd.out_image_dims = (512 * 4, 512 * 4)
                        cmd.out_image_layout = (1, 1)
                        cmd.status = 2 # success
                                       
        except Exception as e:
            cmd.status = -1 # error
            cmd.err_txt = str(e)
            
    else:
        
        cmd.status = -1 # error
        cmd.err_txt = "Unrecognized command"
        
    next_cmd = CMD_QUEUE.get_next_pending()
    
    if cmd.status == 2: # completed successfully
    
        # update the per user total run-time cache
        cmd.elapsed_time = datetime.datetime.now() - cmd.start_time
        if cmd.init_user in USERS_ELAPSED_TIME.keys():
            USERS_ELAPSED_TIME[cmd.init_user] += cmd.elapsed_time
        else:
            USERS_ELAPSED_TIME[cmd.init_user] = cmd.elapsed_time

        # send requesting user their output image
        try:
            file = discord.File(cmd.out_image)
            msg = "Finished @" + cmd.init_user + "  " + cmd.get_summary()
            if next_cmd == None: msg += " - Queue is empty!"
            await cmd.ctx.send(file=file, content=msg)
        except Exception as e:
            cmd.status = -1 # error
            cmd.err_txt = "Error sending output image - " + str(e)
            
    if cmd.status == -1: # error
        
        try:
            msg = "Sorry @" + cmd.init_user + ", " + cmd.err_txt + " - " + cmd.get_summary()
            if next_cmd == None: msg += " - Queue is empty!"
            await cmd.ctx.send(msg)
        except Exception as e:
            print("Error sending error message to user - " + str(e))
    
    # write command file to disk if we have a data file path
    
    CMD_QUEUE.save()
    
    return

@client.command()
async def t2i(ctx):
    global CMD_QUEUE
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    await CMD_QUEUE.add_new(ctx)

@client.command()
async def t2ihd(ctx):
    global CMD_QUEUE
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return    
    await CMD_QUEUE.add_new(ctx)
    
@client.command()
async def i2i(ctx):
    global CMD_QUEUE
    global BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME
    if not _check_server_roles(ctx, [BOT_ADMIN_ROLE_NAME, BOT_USERS_ROLE_NAME]): return
    await CMD_QUEUE.add_new(ctx)
    
@client.command()
async def img2img(ctx):
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
    
    if message.author.bot: return
    if message.content.startswith("!") != True: return
    
    await client.process_commands(message)

@client.event
async def on_disconnect():
    await asyncio.sleep(10)

CMD_QUEUE = CommandQueue(data_file=BOT_STATE_DATA_FILE)
client.run(BOT_TOKEN, reconnect=True)