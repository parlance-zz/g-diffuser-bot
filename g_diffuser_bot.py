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

import extensions.g_diffuser_lib as gdl
from g_diffuser_config import DEFAULT_PATHS, DISCORD_BOT_SETTINGS, GRPC_SERVER_SETTINGS
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

# redirect default paths to designated bot root path
DEFAULT_PATHS.inputs = DEFAULT_PATHS.bot+"/inputs"
DEFAULT_PATHS.outputs = DEFAULT_PATHS.bot+"/outputs"
DEFAULT_PATHS.backups = DEFAULT_PATHS.bot+"/backups"
DEFAULT_PATHS.saved = DEFAULT_PATHS.bot+"/saved"

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
        #app_commands = await self.tree.fetch_commands()
        #for app_command in app_commands:
        #    await app_command.edit(dm_permission=True)

        # explicitly sync all commands with all guilds
        bot_guilds = [discord.Object(guild) for guild in self.settings.guilds]
        for guild in bot_guilds:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
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
        default=DISCORD_BOT_SETTINGS.token,
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
    if (not DISCORD_BOT_SETTINGS.token) or (DISCORD_BOT_SETTINGS.token == "YOUR_DISCORD_BOT_TOKEN_HERE"):
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
    name="dream",
    description="create something",
    nsfw=(GRPC_SERVER_SETTINGS.nsfw_behaviour != "block"),
)
@app_commands.guild_only()
@app_commands.guilds(*DISCORD_BOT_SETTINGS.guilds)
@app_commands.describe(
    prompt='what do you want to create today?',
    model_name='which model to use',
    sampler='which sampling algorithm to use',
    width='width of each output image',
    height='height of each output image',
    scale='conditional guidance scale',
    seed='seed for the random generator',
    steps='number of sampling steps',
    n='number of images to generate at once',
)
@app_commands.choices(
    sampler=SAMPLER_CHOICES,
    model_name=MODEL_CHOICES,
)
async def dream(
    interaction: discord.Interaction,
    prompt: str,# = DEFAULT_SAMPLE_SETTINGS.prompt,
    model_name: Optional[app_commands.Choice[str]] = DEFAULT_SAMPLE_SETTINGS.model_name,
    sampler: Optional[app_commands.Choice[str]] = DEFAULT_SAMPLE_SETTINGS.sampler,
    width: Optional[app_commands.Range[int, 64, DEFAULT_SAMPLE_SETTINGS.max_resolution[0]]] = DEFAULT_SAMPLE_SETTINGS.resolution[0],
    height: Optional[app_commands.Range[int, 64, DEFAULT_SAMPLE_SETTINGS.max_resolution[1]]] = DEFAULT_SAMPLE_SETTINGS.resolution[1],
    scale: Optional[app_commands.Range[float, 0.0, 100.0]] = DEFAULT_SAMPLE_SETTINGS.scale,
    seed: Optional[app_commands.Range[int, 1, 2000000000]] = 0,
    steps: Optional[app_commands.Range[int, 1, DISCORD_BOT_SETTINGS.max_steps_limit]] = DEFAULT_SAMPLE_SETTINGS.steps,
    n: Optional[app_commands.Range[int, 1, DISCORD_BOT_SETTINGS.max_output_limit]] = DISCORD_BOT_SETTINGS.default_output_n,
):
    global DEFAULT_PATHS, DEFAULT_SAMPLE_SETTINGS, GRPC_SERVER_LOCK

    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: pass #print("exception in await interaction - " + str(e))
    
    if not prompt:
        try: await interaction.followup.send(content="sorry @"+interaction.user.display_name+", please enter a prompt", ephemeral=True)
        except Exception as e: pass #print("exception in await interaction - " + str(e))
        return
    
    # build sample args from app command params
    args = gdl.get_default_args()
    args.prompt = prompt
    if type(model_name) == str: args.model_name = model_name
    else: args.model_name = model_name.value
    if type(sampler) == str: args.sampler = sampler
    else: args.sampler = sampler.value
    args.w = width
    args.h = height
    args.scale = scale
    args.seed = seed
    args.steps = steps
    args.n = n
    args.interactive = True
    gdl.print_namespace(args, debug=0, verbosity_level=1)

    try:
        await GRPC_SERVER_LOCK.acquire()
        start_time = datetime.datetime.now()
        #thread = Thread(target = gdl.get_samples, args=[args],  kwargs={'no_grid': True}, daemon=True)
        def get_samples_wrapper(args):
            samples, sample_files = gdl.get_samples(args, no_grid=True)
            threading.current_thread().sample_files = sample_files
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
        if args.n == DEFAULT_SAMPLE_SETTINGS.n: del args.n
        if args.width == DEFAULT_SAMPLE_SETTINGS.resolution[0]: del args.width
        if args.height == DEFAULT_SAMPLE_SETTINGS.resolution[1]: del args.height
        
        # construct args string for echo / acknowledgement
        args_dict = vars(gdl.strip_args(args, level=1))
        args_str = str(args_dict).replace("{","").replace("}","").replace('"', "").replace("'", "").replace(",", " ")
        args_str = "prompt: " + args_prompt + "  " + args_str + "  seed: " + str(args_seed)
        message = "@" + interaction.user.display_name + ":  /dream "+ args_str

        try: await interaction.followup.send(files=attachment_files, content=message)
        except Exception as e: pass # print("exception in await interaction - " + str(e))
    else:
        print("error - " + args.err_txt); gdl.print_namespace(args, debug=1)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: pass # print("exception in await interaction - " + str(e))
        return

    print("elapsed time: " + str(datetime.datetime.now() - start_time) + "s")
    return
    
def get_file_extension_from_url(url):
    tokens = os.path.splitext(os.path.basename(urllib.parse.urlsplit(url).path))
    if len(tokens) > 1: return tokens[1]
    else: return ""

async def download_attachment(url, user, attachment_name):
    global DEFAULT_PATHS
    try:
        sanitized_username = gdl.get_default_output_name(user); sanitized_attachment_name = gdl.get_default_output_name(attachment_name)
        download_path = DEFAULT_PATHS.inputs+"/"+sanitized_username+"/"+sanitized_attachment_name+get_file_extension_from_url(url)
        print("Downloading '" + url + "' to '" + download_path + "'...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(response.content_type)
                breakpoint()
                if response.status == 200:
                    with open(download_path, "wb") as out_file:
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

"""
def _restart_program():
    global CMD_QUEUE
    SCRIPT_FILE_NAME = os.path.basename(__file__)
    print("Restarting...")
    run_string = 'python "'+ SCRIPT_FILE_NAME + '"'
    print(run_string)
    subprocess.Popen(run_string)
    exit(0)

async def _top(ctx):    # replies to a message with a sorted list of all users and their run-time
    global CMD_QUEUE
    i = 0 ; msg = "Okay @" + str(ctx.message.author.name) + ", here's the top users... \n"
    for user in sorted(CMD_QUEUE.users_elapsed_time, reverse=True, key=CMD_QUEUE.users_elapsed_time.get):
        i += 1 ; msg += str(i) + ": @" + user + " <" + str(datetime.timedelta(seconds=CMD_QUEUE.users_elapsed_time[user].seconds)) + "s>\n"
    if i == 0: msg = "No users yet!"
    try: await ctx.send("@" + str(ctx.message.author.name) + " : " + msg)
    except Exception as e: print("Error sending !top acknowledgement - " + str(e))
    return
"""