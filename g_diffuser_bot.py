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

"""

import modules.g_diffuser_lib as gdl
gdl.load_config()

import os; os.chdir(gdl.DEFAULT_PATHS.root)

import sys
import datetime
import pathlib
import yaml
from typing import Optional
import aiohttp
import argparse
from argparse import Namespace

import discord
from discord import app_commands

# redirect default paths to the designated bot path
gdl.DEFAULT_PATHS.inputs = gdl.DEFAULT_PATHS.bot+"/inputs"
gdl.DEFAULT_PATHS.outputs = gdl.DEFAULT_PATHS.bot+"/outputs"
gdl.DEFAULT_PATHS.temp = gdl.DEFAULT_PATHS.bot+"/temp"

gdl.start_grpc_server()
models = gdl.get_models()

MODEL_CHOICES = []
for model in models:
    MODEL_CHOICES.append(app_commands.Choice(name=model["id"], value=model["id"]))

SAMPLER_CHOICES = []
for sampler in gdl.GRPC_SERVER_SUPPORTED_SAMPLERS_LIST:
    SAMPLER_CHOICES.append(app_commands.Choice(name=sampler, value=sampler))

class G_DiffuserBot(discord.Client):
    def __init__(self):
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
        if gdl.DISCORD_BOT_SETTINGS.state_file_path: # load existing data if we have state file path
            try: self.load_state()
            except Exception as e:
                #print("Error loading '{0}' - {1}".format(gdl.DISCORD_BOT_SETTINGS.state_file_path, str(e)))
                pass

        return

    async def setup_hook(self):
        # this is broken, for some reason fetch_commands() always returns nothing
        app_commands = await self.tree.fetch_commands()
        for app_command in app_commands:
            await app_command.edit(dm_permission=True)

        # explicitly sync all commands with all guilds / servers bot has joined
        bot_guilds = client.guilds
        print("Synchronizing app commands with servers/guilds: {0}...\n".format(str([vars(x) for x in bot_guilds])))
        for guild in bot_guilds:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        print("Bot app-command tree: {0}\n\n".format(str(vars(self.tree))))
        return

    def load_state(self):
        with open(gdl.DISCORD_BOT_SETTINGS.state_file_path, 'r') as dfile:
            self.saved_state = argparse.Namespace(**json.load(dfile))
        print("Loaded {0}...".format(gdl.DISCORD_BOT_SETTINGS.state_file_path))
        return

    def save_state(self):
        try:
            (pathlib.Path(gdl.DISCORD_BOT_SETTINGS.state_file_path).parents[0]).mkdir(exist_ok=True, parents=True)
            with open(gdl.DISCORD_BOT_SETTINGS.state_file_path, "w") as dfile:
                json.dump(self.saved_state, dfile)
            print("Saved {0}...".format(gdl.DISCORD_BOT_SETTINGS.state_file_path))
        except Exception as e:
            raise("Error saving '{0}' - {1}".format(gdl.DISCORD_BOT_SETTINGS.state_file_path, str(e)))
        return


class DiscordBotLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        try:
            self.log = open(log_path, "a") # append to existing log file
        except:
            self.log = None
        return
    def __del__(self):
        sys.stdout = self.terminal
        if self.log: self.log.close()
        return
    def write(self, message):
        self.terminal.write(message)
        if self.log: self.log.write(message)
        return
    def flush(self):
        if self.log: self.log.flush()
        return

if __name__ == "__main__":
    sys.stdout = DiscordBotLogger("g_diffuser_bot.log")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--token",
        type=str,
        default=gdl.DISCORD_BOT_SETTINGS.token,
        help="if you want to override the discord bot token in config you can supply an alternate here",
    )
    args = parser.parse_args()
    gdl.DISCORD_BOT_SETTINGS.token = args.token
    # if we don't have a valid discord bot token let's not go any further
    if (not gdl.DISCORD_BOT_SETTINGS.token) or (gdl.DISCORD_BOT_SETTINGS.token == "{your discord bot token}"):
        print("Fatal error: Cannot start discord bot with token '{0}'".format(gdl.DISCORD_BOT_SETTINGS.token))
        print("Please update DISCORD_BOT_TOKEN in config. Press enter to continue...")
        input(); exit(1)
    else:
        client = G_DiffuserBot()


@client.tree.command(
    name="img",
    description="use an input image for img2img, inpainting, or outpainting",
#    nsfw=(gdl.GRPC_SERVER_SETTINGS.nsfw_behaviour != "block"),
)
@app_commands.describe(
    input_image_url="url of input image (you can right-click an image in discord chat to get a link)",
    prompt='what do you want to create today?',
    num_samples='number of images to generate at once',
    model_name='which model to use',
    sampler='which sampling algorithm to use',
    cfg_scale='classifier-free guidance scale',
    seed='seed for the random generator',
    steps='number of sampling steps',
    negative_prompt='has the effect of an anti-prompt',
    guidance_strength='clip guidance (only affects clip models)',
    input_image_url='optional input image url for in/out-painting or img2img',
    img2img_strength='amount to change the input image (only affects img2img, not in/out-painting)',
    expand_top='expand input image top by how much (in %)?',
    expand_right='expand input image right by how much (in %)?',
    expand_bottom='expand input image bottom by how much (in %)?',
    expand_left='expand input image left by how much (in %)?',
    expand_softness='amount to soften the resulting input image mask (in %)',
    expand_space='distance erased from the edge of the original input image',    
)
@app_commands.choices(
    sampler=SAMPLER_CHOICES,
    model_name=MODEL_CHOICES,
)
async def img(
    interaction: discord.Interaction,
    input_image_url: str,
    prompt: Optional[str]= "",
    num_samples: Optional[app_commands.Range[int, 1, gdl.DISCORD_BOT_SETTINGS.max_output_limit]] = gdl.DISCORD_BOT_SETTINGS.default_output_n,
    model_name: Optional[app_commands.Choice[str]] = gdl.DEFAULT_SAMPLE_SETTINGS.model_name,
    sampler: Optional[app_commands.Choice[str]] = gdl.DEFAULT_SAMPLE_SETTINGS.sampler,
    cfg_scale: Optional[app_commands.Range[float, 0.0, 100.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.cfg_scale,
    seed: Optional[app_commands.Range[int, 1, 2000000000]] = 0,
    steps: Optional[app_commands.Range[int, 1, gdl.DISCORD_BOT_SETTINGS.max_steps_limit]] = gdl.DEFAULT_SAMPLE_SETTINGS.steps,
    negative_prompt: Optional[str] = gdl.DEFAULT_SAMPLE_SETTINGS.negative_prompt,
    guidance_strength: Optional[app_commands.Range[float, 0.0, 1.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.guidance_strength,
    img2img_strength: Optional[app_commands.Range[float, 0.0, 2.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.img2img_strength,
    expand_top: Optional[app_commands.Range[float, 0.0, 1000.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_top,
    expand_bottom: Optional[app_commands.Range[float, 0.0, 1000.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_bottom,
    expand_left: Optional[app_commands.Range[float, 0.0, 1000.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_left,
    expand_right: Optional[app_commands.Range[float, 0.0, 1000.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_right,
    expand_softness: Optional[app_commands.Range[float, 0.0, 1000.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_softness,
    expand_space: Optional[app_commands.Range[float, 0., 200.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_space,
    
):
    # build sample args from app command params
    args = locals().copy()
    del args.interaction
    args["model_name"] = args["model_name"].value
    args["sampler"] = args["sampler"].value
    
    args = Namespace(**(vars(gdl.get_default_args()) | args))
    init_image = await download_attachment(input_image_url)
    args.init_image = init_image
    gdl.print_args(args, verbosity_level=1)

    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: print("exception in await interaction - " + str(e))

    # ...

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
        print("error - " + args.err_txt); gdl.print_args(args)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))
        return

    print("elapsed time: " + str(datetime.datetime.now() - start_time) + "s")
    return

@client.tree.command(
    name="g",
    description="create something",
#    nsfw=(gdl.GRPC_SERVER_SETTINGS.nsfw_behaviour != "block"),
)
@app_commands.describe(
    prompt='what do you want to create today? use a single space for no prompt',
    n='number of images to generate at once',
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
    top='expand input image top by how much (in %)?',
    right='expand input image right by how much (in %)?',
    bottom='expand input image bottom by how much (in %)?',
    left='expand input image left by how much (in %)?',
    softness='amount to soften the resulting input image mask (in %)',
    space='distance erased from the edge of the original input image',    
)
@app_commands.choices(
    sampler=SAMPLER_CHOICES,
    model_name=MODEL_CHOICES,
)
async def g(
    interaction: discord.Interaction,
    prompt: str,
    n: Optional[app_commands.Range[int, 1, gdl.DISCORD_BOT_SETTINGS.max_output_limit]] = gdl.DISCORD_BOT_SETTINGS.default_output_n,
    model_name: Optional[app_commands.Choice[str]] = gdl.DEFAULT_SAMPLE_SETTINGS.model_name,
    sampler: Optional[app_commands.Choice[str]] = gdl.DEFAULT_SAMPLE_SETTINGS.sampler,
    width: Optional[app_commands.Range[int, 64, gdl.DEFAULT_SAMPLE_SETTINGS.max_resolution[0]]] = gdl.DEFAULT_SAMPLE_SETTINGS.resolution[0],
    height: Optional[app_commands.Range[int, 64, gdl.DEFAULT_SAMPLE_SETTINGS.max_resolution[1]]] = gdl.DEFAULT_SAMPLE_SETTINGS.resolution[1],
    scale: Optional[app_commands.Range[float, 0.0, 100.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.scale,
    seed: Optional[app_commands.Range[int, 1, 2000000000]] = 0,
    steps: Optional[app_commands.Range[int, 1, gdl.DISCORD_BOT_SETTINGS.max_steps_limit]] = gdl.DEFAULT_SAMPLE_SETTINGS.steps,
    negative_prompt: Optional[str] = gdl.DEFAULT_SAMPLE_SETTINGS.negative_prompt,
    guidance_strength: Optional[app_commands.Range[float, 0.0, 1.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.guidance_strength,
    input_image_url: Optional[str] = "",
    img2img_strength: Optional[app_commands.Range[float, 0.0, 2.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.noise_start,
    top: Optional[app_commands.Range[float, 0.0, 1000.0]] = 0.,
    right: Optional[app_commands.Range[float, 0.0, 1000.0]] = 0.,
    bottom: Optional[app_commands.Range[float, 0.0, 1000.0]] = 0.,
    left: Optional[app_commands.Range[float, 0.0, 1000.0]] = 0.,
    softness: Optional[app_commands.Range[float, 0.0, 1000.0]] = 100.,
    space: Optional[app_commands.Range[float, 0.1, 100.0]] = 1.,
    
):
    global GRPC_SERVER_LOCK

    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: print("exception in await interaction - " + str(e))

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
    gdl.print_args(args, verbosity_level=1)

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
        print("error - " + str(e)); gdl.print_args(args)
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
        print("error - " + args.err_txt); gdl.print_args(args)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))
        return

    print("elapsed time: " + str(datetime.datetime.now() - start_time) + "s")
    return

async def download_attachment(url):   
    mime_types ={
        "image/png" : ".png",
        "image/jpeg" : ".jpg",
        "image/gif" : ".gif",
        "image/bmp" : ".bmp",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                attachment_type = response.content_type
                if attachment_type not in mime_types:
                    raise Exception("attachment type '{0}' not found in allowed attachment list '{1}'".format(attachment_type, mime_types))

                if response.status == 200:
                    attachment_extension = mime_types[attachment_type]
                    sanitized_attachment_name = gdl.get_default_output_name(url)
                    download_path = sanitized_attachment_name+attachment_extension
                    full_download_path = gdl.DEFAULT_PATHS.inputs+"/"+download_path
                    print("Downloading '" + url + "' to '" + full_download_path + "'...")
                    with open(full_download_path, "wb") as out_file:
                        out_file.write(await response.read())
                else:
                    raise("Error downloading url, status = {0}".format(str(response.status)))
        
    except Exception as e:
        raise("Error downloading url - {0}".format(str(e)))
        
    return download_path

if __name__ == "__main__":
    client.run(gdl.DISCORD_BOT_SETTINGS.token, reconnect=True)