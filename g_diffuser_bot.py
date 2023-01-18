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
from modules.g_diffuser_lib import SimpleLogger

if __name__ == "__main__":
    gdl.load_config()
    import os; os.chdir(gdl.DEFAULT_PATHS.root)
    logger = SimpleLogger("g_diffuser_bot.log")
    
import datetime
import pathlib
import yaml
from typing import Optional
import aiohttp
import argparse
from argparse import Namespace
import inspect

import discord
from discord import app_commands

# redirect default paths to the designated bot path
gdl.DEFAULT_PATHS.inputs = gdl.DEFAULT_PATHS.bot+"/inputs"
gdl.DEFAULT_PATHS.outputs = gdl.DEFAULT_PATHS.bot+"/outputs"
gdl.DEFAULT_PATHS.temp = gdl.DEFAULT_PATHS.bot+"/temp"

models = gdl.start_grpc_server()
if models == None:
    raise Exception("Error: SDGRPC server is unavailable")
    
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

    async def setup_commands(self):
        # this is broken, for some reason fetch_commands() always returns nothing
        app_commands = await self.tree.fetch_commands()
        for app_command in app_commands:
            await app_command.edit(dm_permission=True)

        # explicitly sync all commands with all guilds / servers bot has joined
        bot_guilds = client.guilds
        print("Synchronizing app commands with servers/guilds: {0}...\n".format(str([str(x.id) for x in bot_guilds])))
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


if __name__ == "__main__":
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


def check_discord_attachment_size(file_path):
    return file_path # todo: check if img is too large, if so convert and save to jpg, return that path instead

def get_discord_echo_args(args, img2img_params=False):
    if args.seed: args.seed -= 1 # show the seed that was _used_, not the next seed
    if args.auto_seed: args.auto_seed -= 1

    if img2img_params:
        if "width" in args: del args.width
        if "height" in args: del args.height

    if args.prompt == "":
        del args.prompt

    args_string = gdl.print_args(args, verbosity_level=2, return_only=True, width=9999)
    return args_string.replace("\n", "\t")

@client.event
async def on_ready():
    await client.setup_commands()

# img command for generating with an input image (img2img, in/outpainting)
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
    img2img_strength='amount to change the input image (only affects img2img, not in/out-painting)',
    expand_top='expand input image top by how much (in %)?',
    expand_right='expand input image right by how much (in %)?',
    expand_bottom='expand input image bottom by how much (in %)?',
    expand_left='expand input image left by how much (in %)?',
    expand_all='expand input image in _every_ direction by how much (in %)?',
    expand_softness='amount to soften the resulting input image mask (in %)',
    expand_space='distance erased from the edge of the original input image',
    hires_fix='Use the hires fix system to improve quality of large images',
    seamless_tiling='Generate a seamlessly tileable image',
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
    expand_all: Optional[app_commands.Range[float, 0.0, 1000.0]] = 0.,
    expand_softness: Optional[app_commands.Range[float, 0.0, 100.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_softness,
    expand_space: Optional[app_commands.Range[float, 0., 100.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.expand_space,
    hires_fix: Optional[bool] = gdl.DEFAULT_SAMPLE_SETTINGS.hires_fix,
    seamless_tiling: Optional[bool] = gdl.DEFAULT_SAMPLE_SETTINGS.seamless_tiling,
):
    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: print("exception in await interaction - " + str(e))

    # build sample args from app command params
    args = locals().copy()
    del args["interaction"]

    args["expand_top"] += args["expand_all"]
    args["expand_bottom"] += args["expand_all"]
    args["expand_left"] += args["expand_all"]
    args["expand_right"] += args["expand_all"]
    del args["expand_all"]

    if type(args["model_name"]) != str: args["model_name"] = args["model_name"].value
    if type(args["sampler"]) != str: args["sampler"] = args["sampler"].value
    
    args = Namespace(**(vars(gdl.get_default_args()) | args))
    init_image = await download_attachment(input_image_url)
    args.init_image = init_image
    gdl.print_args(args)

    output_args = await gdl.get_samples(args)

    if args.status == 2: # completed successfully
        attachment_files = []
        for output in output_args:
            sample_filename = check_discord_attachment_size(gdl.DEFAULT_PATHS.outputs + "/" + output.output_file)
            attachment_files.append(discord.File(sample_filename))
        
        args_str = get_discord_echo_args(args, img2img_params=True)
        cmd_str = inspect.currentframe().f_code.co_name
        message = "@" + interaction.user.display_name + f":  /{cmd_str} {args_str}"

        try: await interaction.followup.send(files=attachment_files, content=message)
        except Exception as e: print("exception in await interaction - " + str(e))
    else:
        print("error - " + args.error_message); gdl.print_args(args, verbosity_level=0)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))

    print("")
    return

# txt2img command
@client.tree.command(
    name="g",
    description="create something",
#    nsfw=(gdl.GRPC_SERVER_SETTINGS.nsfw_behaviour != "block"),
)
@app_commands.describe(
    prompt='what do you want to create today?',
    num_samples='number of images to generate at once',
    model_name='which model to use',
    sampler='which sampling algorithm to use',
    cfg_scale='classifier-free guidance scale',
    seed='seed for the random generator',
    steps='number of sampling steps',
    negative_prompt='has the effect of an anti-prompt',
    guidance_strength='clip guidance (only affects clip models)',
    width='width (in pixels) of the output image',
    height='height (in pixels) of the output image',
    hires_fix='Use the hires fix system to improve quality of large images',
    seamless_tiling='Generate a seamlessly tileable image',    
)
@app_commands.choices(
    sampler=SAMPLER_CHOICES,
    model_name=MODEL_CHOICES,
)
async def g(
    interaction: discord.Interaction,
    prompt: str,
    num_samples: Optional[app_commands.Range[int, 1, gdl.DISCORD_BOT_SETTINGS.max_output_limit]] = gdl.DISCORD_BOT_SETTINGS.default_output_n,
    model_name: Optional[app_commands.Choice[str]] = gdl.DEFAULT_SAMPLE_SETTINGS.model_name,
    sampler: Optional[app_commands.Choice[str]] = gdl.DEFAULT_SAMPLE_SETTINGS.sampler,
    cfg_scale: Optional[app_commands.Range[float, 0.0, 100.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.cfg_scale,
    seed: Optional[app_commands.Range[int, 1, 2000000000]] = 0,
    steps: Optional[app_commands.Range[int, 1, gdl.DISCORD_BOT_SETTINGS.max_steps_limit]] = gdl.DEFAULT_SAMPLE_SETTINGS.steps,
    negative_prompt: Optional[str] = gdl.DEFAULT_SAMPLE_SETTINGS.negative_prompt,
    guidance_strength: Optional[app_commands.Range[float, 0.0, 1.0]] = gdl.DEFAULT_SAMPLE_SETTINGS.guidance_strength,
    width: Optional[app_commands.Range[int, 512, gdl.DEFAULT_SAMPLE_SETTINGS.max_width]] = gdl.DEFAULT_SAMPLE_SETTINGS.width,
    height: Optional[app_commands.Range[int, 512, gdl.DEFAULT_SAMPLE_SETTINGS.max_height]] = gdl.DEFAULT_SAMPLE_SETTINGS.height,
    hires_fix: Optional[bool] = gdl.DEFAULT_SAMPLE_SETTINGS.hires_fix,
    seamless_tiling: Optional[bool] = gdl.DEFAULT_SAMPLE_SETTINGS.seamless_tiling,    
):
    try: await interaction.response.defer(thinking=True, ephemeral=False) # start by requesting more time to respond
    except Exception as e: print("exception in await interaction - " + str(e))

    # build sample args from app command params
    args = locals().copy()
    del args["interaction"]
    if type(args["model_name"]) != str: args["model_name"] = args["model_name"].value
    if type(args["sampler"]) != str: args["sampler"] = args["sampler"].value
    
    args = Namespace(**(vars(gdl.get_default_args()) | args))
    gdl.print_args(args)

    output_args = await gdl.get_samples(args)

    if args.status == 2: # completed successfully
        attachment_files = []
        for output in output_args:
            sample_filename = check_discord_attachment_size(gdl.DEFAULT_PATHS.outputs + "/" + output.output_file)
            attachment_files.append(discord.File(sample_filename))
        
        args_str = get_discord_echo_args(args)
        cmd_str = inspect.currentframe().f_code.co_name
        message = "@" + interaction.user.display_name + f":  /{cmd_str} {args_str}"

        try: await interaction.followup.send(files=attachment_files, content=message)
        except Exception as e: print("exception in await interaction - " + str(e))
    else:
        print("error - " + args.error_message); gdl.print_args(args, verbosity_level=0)
        try: await interaction.followup.send(content="sorry, something went wrong :(", ephemeral=True)
        except Exception as e: print("exception in await interaction - " + str(e))

    print("")
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