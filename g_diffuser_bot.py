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


g_diffuser_bot.py - discord bot interface for g-diffuser-lib
                  - Oct 2022 Update: I am in the process of rebuilding this bot from the ground up on top of the
                  the new g-diffuser-lib code / framework. I intend to add all the CLI functions, img2img, outpainting,
                  GUI buttons and much more but will need time to put it all back together. Thank you.

"""

import modules.g_diffuser_lib as gdl
from g_diffuser_config import DEFAULT_PATHS, DISCORD_BOT_SETTINGS
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS

import os; os.chdir(DEFAULT_PATHS.root)

import datetime
import pathlib
import urllib
import json
from typing import Optional
#import glob
import aiohttp
#import datetime
import argparse
import threading
from threading import Thread
import asyncio

import discord
#from discord.ext import commands
#from discord.ext import tasks
from discord import app_commands

import numpy as np
import cv2

# redirect default paths to designated bot root path
DEFAULT_PATHS.inputs = DEFAULT_PATHS.bot+"/inputs"
DEFAULT_PATHS.outputs = DEFAULT_PATHS.bot+"/outputs"

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")

# help and about strings, these must be 2000 characters or less
ABOUT_TXT = """
"""
HELP_TXT = """
"""
EXAMPLES_TXT = """
"""

SAMPLER_CHOICES = []
for sampler in gdl.SUPPORTED_SAMPLERS_LIST:
    SAMPLER_CHOICES.append(app_commands.Choice(name=sampler, value=sampler))

MODEL_CHOICES = []
for model in DISCORD_BOT_SETTINGS.model_list:
    MODEL_CHOICES.append(app_commands.Choice(name=model, value=model))

GRPC_SERVER_LOCK = asyncio.Lock()

class G_DiffuserBot(discord.Client):
    def __init__(self):
        global DISCORD_BOT_SETTINGS
        self.settings = DISCORD_BOT_SETTINGS
        intents = discord.Intents(
            messages=True,
            dm_messages=True,
            guild_messages=True,
            message_content=True,
        )
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

        self.restart_now = None
        self.shutdown_now = None
        self.cmd_list = []

        self.saved_state = argparse.Namespace()
        self.saved_state.users_total_elapsed_time = {}
        if DISCORD_BOT_SETTINGS.state_file_path: # load existing data if we have state file path
            try: self.load_state()
            except Exception as e:
                #print("Error loading '"+DISCORD_BOT_SETTINGS.state_file_path+"' - "+str(e))
                pass

        gdl.start_grpc_server(gdl.get_default_args())
        return

    async def setup_hook(self):
        # this is broken, for some reason fetch_commands() always returns nothing
        app_commands = await self.tree.fetch_commands()
        for app_command in app_commands:
            await app_command.edit(dm_permission=True)

        # explicitly sync all commands with all guilds
        bot_guilds = [discord.Object(guild) for guild in self.settings.guilds]
        print("Synchronizing app commands with servers/guilds: " + str([vars(x) for x in bot_guilds]) + "...\n")
        for guild in bot_guilds:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        print("Bot app-command tree: " + str(vars(self.tree)) + "\n\n")
        return

    def load_state(self):
        with open(self.settings.state_file_path, 'r') as dfile:
            self.saved_state = argparse.Namespace(**json.load(dfile))
            dfile.close()
        print("Loaded " + self.settings.state_file_path + "...")
        return

    def save_state(self):
        try:
            (pathlib.Path(self.settings.state_file_path).parents[0]).mkdir(exist_ok=True, parents=True)
            with open(self.settings.state_file_path, "w") as dfile:
                json.dump(self.saved_state, dfile)
                dfile.close()
            print("Saved " + self.settings.state_file_path + "...")
        except Exception as e:
            print("Error saving '" + self.settings.state_file_path + "' - " + str(e))
        return
                
    async def add_new(self, ctx): # add a new command to the queue
        rejected = (self.restart_now != None) or (self.shutdown_now != None)
        rejected |= (len(self.cmd_list) >= self.settings.max_queue_length)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--token",
        type=str,
        default=DISCORD_BOT_TOKEN,
        help="if you want to override the discord bot token in g_diffuser_config.py you can supply an alternate here",
    )
    parser.add_argument(
        "--guilds",
        type=int,
        nargs="+",
        default=DISCORD_BOT_SETTINGS.guilds,
        help="if you want to override the guild id(s) in g_diffuser_config.py you can supply alternate(s) here",
    )
    args = parser.parse_args()
    DISCORD_BOT_SETTINGS.token = args.token
    DISCORD_BOT_SETTINGS.guilds = args.guilds
    # if we don't have a valid discord bot token or guild id let's not go any further
    if (not DISCORD_BOT_SETTINGS.token) or (DISCORD_BOT_SETTINGS.token == "{your discord bot token}"):
        print("Fatal error: Cannot start discord bot with token '" + DISCORD_BOT_SETTINGS.token + "'")
        print("Please update DISCORD_BOT_SETTINGS.token in g_diffuser_config.py and try again.")
        exit(1)
    else:
        if len(DISCORD_BOT_SETTINGS.guilds) < 1:
            print("Fatal error: Cannot start discord bot with no guild ids")
            print("Please update DISCORD_BOT_SETTINGS.guilds in g_diffuser_config.py and try again.")
            exit(1)
        else:
            client = G_DiffuserBot()

@client.tree.command(
    name="expand",
    description="resize an input image canvas, filling the new area with transparency",
)
@app_commands.describe(
    top='expand top by how much (in %)?',
    right='expand right by how much (in %)?',
    bottom='expand bottom by how much (in %)?',
    left='expand left by how much (in %)?',
    input_image_url='input image url for expansion',
    softness='amount to soften the resulting mask (in %)',
    space='distance erased from the edge of the original image',
)
async def expand(
    interaction: discord.Interaction,
    input_image_url: str,
    top: Optional[app_commands.Range[float, 0.0, 1000.0]] = 25.,
    right: Optional[app_commands.Range[float, 0.0, 1000.0]] = 25.,
    bottom: Optional[app_commands.Range[float, 0.0, 1000.0]] = 25.,
    left: Optional[app_commands.Range[float, 0.0, 1000.0]] = 25.,
    softness: Optional[app_commands.Range[float, 0.0, 1000.0]] = 85.,
    space: Optional[app_commands.Range[float, 0.1, 100.0]] = 1.,
):
    global DEFAULT_PATHS

    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: print("exception in await interaction - " + str(e))
    
    if not input_image_url:
        try: await interaction.followup.send(content="sorry @"+interaction.user.display_name+", please enter an input_image_url", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))
        return

    try:
        init_img = await download_attachment(input_image_url)
        init_img_fullpath = DEFAULT_PATHS.inputs+"/"+init_img
        cv2_img = cv2.imread(init_img_fullpath)

        new_img = gdl.expand_image(cv2_img, top, right, bottom, left, softness, space)
        new_img_fullpath = DEFAULT_PATHS.outputs+"/"+init_img+".expanded.png"
        gdl.save_image(new_img, new_img_fullpath)
        print("Saved " + new_img_fullpath)
        await interaction.followup.send(content="@"+interaction.user.display_name+" - here's your expanded image:", file=discord.File(new_img_fullpath), ephemeral=True)

    except Exception as e:
        print("error - " + str(e))
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))
        return
    return

@client.tree.command(
    name="g",
    description="create something",
#    nsfw=(GRPC_SERVER_SETTINGS.nsfw_behaviour != "block"),
)
@app_commands.describe(
    prompt='what do you want to create today?',
    model_name='which model to use',
    sampler='which sampling algorithm to use',
    width='width of each output image',
    height='height of each output image',
    scale='conditional guidance scale',
    seed='seed for the random generator',
    steps='number of sampling steps',
    negative_prompt='has the effect of an anti-prompt',
    guidance_strength='clip guidance (only affects clip models)',
    input_image_url='optional input image url for in/out-painting or img2img',
    img2img_strength='amount to change the input image (only affects img2img, not in/out-painting)',
    n='number of images to generate at once',
)
@app_commands.choices(
    sampler=SAMPLER_CHOICES,
    model_name=MODEL_CHOICES,
)
async def g(
    interaction: discord.Interaction,
    #prompt: str,
    prompt: Optional[str] = "",
    model_name: Optional[app_commands.Choice[str]] = DEFAULT_SAMPLE_SETTINGS.model_name,
    sampler: Optional[app_commands.Choice[str]] = DEFAULT_SAMPLE_SETTINGS.sampler,
    width: Optional[app_commands.Range[int, 64, DEFAULT_SAMPLE_SETTINGS.max_resolution[0]]] = DEFAULT_SAMPLE_SETTINGS.resolution[0],
    height: Optional[app_commands.Range[int, 64, DEFAULT_SAMPLE_SETTINGS.max_resolution[1]]] = DEFAULT_SAMPLE_SETTINGS.resolution[1],
    scale: Optional[app_commands.Range[float, 0.0, 100.0]] = DEFAULT_SAMPLE_SETTINGS.scale,
    seed: Optional[app_commands.Range[int, 1, 2000000000]] = 0,
    steps: Optional[app_commands.Range[int, 1, DISCORD_BOT_SETTINGS.max_steps_limit]] = DEFAULT_SAMPLE_SETTINGS.steps,
    negative_prompt: Optional[str] = DEFAULT_SAMPLE_SETTINGS.negative_prompt,
    guidance_strength: Optional[app_commands.Range[float, 0.0, 1.0]] = DEFAULT_SAMPLE_SETTINGS.guidance_strength,
    input_image_url: Optional[str] = "",
    img2img_strength: Optional[app_commands.Range[float, 0.0, 2.0]] = DEFAULT_SAMPLE_SETTINGS.noise_start,
    n: Optional[app_commands.Range[int, 1, DISCORD_BOT_SETTINGS.max_output_limit]] = DISCORD_BOT_SETTINGS.default_output_n,
):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS, GRPC_SERVER_LOCK

    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: print("exception in await interaction - " + str(e))
    
    if not prompt:
        prompt = " "
        #try: await interaction.followup.send(content="sorry @"+interaction.user.display_name+", please enter a prompt", ephemeral=True)
        #except Exception as e: print("exception in await interaction - " + str(e))
        #return

    if input_image_url != "":
        init_img = await download_attachment(input_image_url)
    else:
        init_img = ""
        
    # build sample args from app command params
    args = gdl.get_default_args()
    args.prompt = prompt
    if type(model_name) == str: args.model_name = model_name
    else: args.model_name = model_name.value
    if type(sampler) == str: args.sampler = sampler
    else: args.sampler = sampler.value
    if width != DEFAULT_SAMPLE_SETTINGS.resolution[0]: args.w = width
    if height != DEFAULT_SAMPLE_SETTINGS.resolution[1]: args.h = height
    args.scale = scale
    args.seed = seed
    args.steps = steps
    args.negative_prompt = negative_prompt
    args.guidance_strength = guidance_strength
    if init_img != "": args.init_img = init_img
    args.noise_start = img2img_strength
    args.n = n
    args.interactive = True
    gdl.print_namespace(args, debug=0, verbosity_level=1)

    try:
        await GRPC_SERVER_LOCK.acquire()
        start_time = datetime.datetime.now()

        def get_samples_wrapper(args):
            try:
                samples, sample_files = gdl.get_samples(args, no_grid=True)
                threading.current_thread().sample_files = sample_files
            except Exception as e:
                threading.current_thread().sample_files = []
                threading.current_thread().err_txt = str(e)
            return

        sample_thread = Thread(target = get_samples_wrapper, args=[args], daemon=True)
        sample_thread.start()
        while True:
            sample_thread.join(0.0001)
            if not sample_thread.is_alive(): break
            await asyncio.sleep(0.05)
        sample_thread.join() # it's the only way to be sure
    except Exception as e:
        print("error - " + str(e)); gdl.print_namespace(args, debug=1)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))
        return
    finally:
        GRPC_SERVER_LOCK.release()

    if "output_file" in args:
        attachment_files = []
        for sample_file in sample_thread.sample_files:
            sample_filename = DEFAULT_PATHS.outputs + "/" + sample_file
            attachment_files.append(discord.File(sample_filename))

        args_prompt = args.prompt; del args.prompt # extract and re-attach the prompt for formatting we can keep apostrophes and commas
        if args.seed != 0: args_seed = args.seed-args.n # only need to show the seed as seed for simplicity
        else: args_seed = args.auto_seed-args.n
        del args.seed
        if "auto_seed" in vars(args): del args.auto_seed

        args_width = args.w; args_height = args.h # replace w and h with width and height for copy/paste and consistency
        del args.w; del args.h
        args.width = args_width
        args.height = args_height

        # don't echo parameters if they have a default value
        if args.model_name == DEFAULT_SAMPLE_SETTINGS.model_name: del args.model_name
        if args.sampler == DEFAULT_SAMPLE_SETTINGS.sampler: del args.sampler
        if args.steps == DEFAULT_SAMPLE_SETTINGS.steps: del args.steps
        if args.scale == DEFAULT_SAMPLE_SETTINGS.scale: del args.scale
        if args.noise_q == DEFAULT_SAMPLE_SETTINGS.noise_q: del args.noise_q
        if args.noise_start == DEFAULT_SAMPLE_SETTINGS.noise_start: del args.noise_start
        if args.noise_end == DEFAULT_SAMPLE_SETTINGS.noise_end: del args.noise_end
        if args.noise_eta == DEFAULT_SAMPLE_SETTINGS.noise_eta: del args.noise_eta
        if args.n == DEFAULT_SAMPLE_SETTINGS.n: del args.n
        if width == DEFAULT_SAMPLE_SETTINGS.resolution[0]: del args.width
        if height == DEFAULT_SAMPLE_SETTINGS.resolution[1]: del args.height
        if args.negative_prompt == DEFAULT_SAMPLE_SETTINGS.negative_prompt: del args.negative_prompt
        if args.guidance_strength == DEFAULT_SAMPLE_SETTINGS.guidance_strength: del args.guidance_strength
        if args.init_img != "":
             del args.init_img
             args.input_image_url = input_image_url
        if "noise_start" in args:
            noise_start = args.noise_start
            del args.noise_start
            if noise_start != DEFAULT_SAMPLE_SETTINGS.min_outpaint_noise:
                args.img2img_strength = noise_start

        # construct args string for echo / acknowledgement
        args_dict = vars(gdl.strip_args(args, level=1))
        args_str = str(args_dict).replace("{","").replace("}","").replace('"', "").replace("'", "").replace(",", " ")
        args_str = "prompt: " + args_prompt + "  " + args_str + "  seed: " + str(args_seed)
        message = "@" + interaction.user.display_name + ":  /g "+ args_str

        try: await interaction.followup.send(files=attachment_files, content=message)
        except Exception as e: print("exception in await interaction - " + str(e))
    else:
        print("error - " + args.err_txt); gdl.print_namespace(args, debug=1)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))
        return

    print("elapsed time: " + str(datetime.datetime.now() - start_time) + "s")
    return
    
def get_file_extension_from_url(url):
    tokens = os.path.splitext(os.path.basename(urllib.parse.urlsplit(url).path))
    if len(tokens) > 1: return tokens[1].strip().lower()
    else: return ""

async def download_attachment(url):
    global DEFAULT_PATHS, DISCORD_BOT_SETTINGS
    try:
        attachment_extension = get_file_extension_from_url(url)
        sanitized_attachment_name = gdl.get_default_output_name(url)
        download_path = sanitized_attachment_name+attachment_extension
        full_download_path = DEFAULT_PATHS.inputs+"/"+download_path

        print("Downloading '" + url + "' to '" + full_download_path + "'...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                attachment_type = response.content_type
                if attachment_type not in DISCORD_BOT_SETTINGS.accepted_attachments:
                    raise Exception("attachment type '"+attachment_type+"' not found in allowed attachment list '"+str(DISCORD_BOT_SETTINGS.accepted_attachments)+"'")
                                 
                if response.status == 200:
                    with open(full_download_path, "wb") as out_file:
                        out_file.write(await response.read())
                        out_file.close()
                else:
                    print("Error downloading url, status = " + str(response.status))
                    return ""
        
    except Exception as e:
        print("Error downloading url - " + str(e))
        return ""
        
    return download_path

if __name__ == "__main__":
    client.run(DISCORD_BOT_SETTINGS.token, reconnect=True)