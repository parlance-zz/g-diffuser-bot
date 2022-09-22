from pathlib import Path
import argparse

DEFAULT_PATHS = argparse.Namespace()
MODEL_DEFAULTS = argparse.Namespace()
DISCORD_BOT_SETTINGS = argparse.Namespace()
CMD_SERVER_SETTINGS = argparse.Namespace()

# ******************** SETTINGS BEGIN ************************

#IMPORTANT - Change default paths here if desired, these are relative to the main root path
DEFAULT_PATHS.models = "./models"
DEFAULT_PATHS.inputs = "./inputs"
DEFAULT_PATHS.outputs = "./outputs"
DEFAULT_PATHS.backups = "./backups"
DEFAULT_PATHS.temp = "./temp"
DEFAULT_PATHS.debug = "./debug"

#IMPORTANT - Default model settings
MODEL_DEFAULTS.model_name = "stable-diffusion-v1-4"  # local path to downloaded model relative to DEFAULT_PATHS.models
#MODEL_DEFAULTS.model_name = "waifu-diffusion"
MODEL_DEFAULTS.use_optimized = False    # set this to True to lower memory consumption (attention slicing and fp16)
#MODEL_DEFAULTS.pipe_list = ["txt2img"] # if you'd like to lower memory consumption even further, you can opt to load only a sub-selection of pipes
#MODEL_DEFAULTS.pipe_list = ["img2img"]
#IMPORTANT - If you want to use a huggingface access token and download models just-in-time, enter your token below
# - Sign up for an account on https://www.huggingface.co and accept the required usage terms for the model of your choice (stable-diffusion-v1-4)
# - Create an access token at https://huggingface.co/settings/tokens and enter your token below:
#MODEL_DEFAULTS.hf_token = "YOUR_HUGGINGFACE_ACCESS_TOKEN_HERE"
#MODEL_DEFAULTS.model_name = "CompVis/stable-diffusion-v1-4"

#IMPORTANT - If you want to use the discord bot, use enter your access token here
# - Discord developers site: https://discordapp.com/developers/applications/
# - Discord bot setup guide: https://www.writebots.com/discord-bot-token/
DISCORD_BOT_SETTINGS.token = "YOUR_DISCORD_BOT_TOKEN_HERE"
DISCORD_BOT_SETTINGS.admin_role = "moderator" #IMPORTANT - use your discord server / guild role names here accordingly, check the bot help text for associated command permissions
DISCORD_BOT_SETTINGS.users_role = "everyone"  # if you want anyone to be able to use the bot, change this to "everyone"
DISCORD_BOT_SETTINGS.cmd_prefix = "!"               # all discord bot commands will require this prefix (e.g. !gen)
DISCORD_BOT_SETTINGS.queue_mode = 0                 # 0 for round-robin, 1 for first come first serve
DISCORD_BOT_SETTINGS.queue_poll_interval = 0.25     # how often should the queue look for new commands to begin processing (in seconds)
DISCORD_BOT_SETTINGS.max_queue_length = 1000        # beyond this limit additional commands will be rejected
DISCORD_BOT_SETTINGS.max_queue_print_items = 4      # max number of items to show for !queue command (up to discord message length limit)
DISCORD_BOT_SETTINGS.max_repeat_limit = 100         # max number of repetitions that can be used with the -x param
DISCORD_BOT_SETTINGS.max_output_limit = 3           # max number of samples to create with -n param
DISCORD_BOT_SETTINGS.max_steps_limit = 100          # max number of steps per sample command
DISCORD_BOT_SETTINGS.default_n = 3                  # default number of samples per command
DISCORD_BOT_SETTINGS.accepted_attachments = [".png", ".jpg", ".jpeg"] # attachments in bot commands not matching this list will not be downloaded
DISCORD_BOT_SETTINGS.state_file_path = "./g-diffuser-bot.json"        # relative to root path
DISCORD_BOT_SETTINGS.activity = DISCORD_BOT_SETTINGS.cmd_prefix + "help, " + DISCORD_BOT_SETTINGS.cmd_prefix + "about"

#IMPORTANT - Command server settings
CMD_SERVER_SETTINGS.http_host = "localhost" # by default the command server binds to localhost to support the discord bot
CMD_SERVER_SETTINGS.http_port = 39132       # change port if needed

# ******************** SETTINGS END ************************

# make global default paths namespace
root_path = Path(__file__).parent.absolute()
_default_paths = {"root": root_path.as_posix()}
DEFAULT_PATHS = vars(DEFAULT_PATHS)
for path in DEFAULT_PATHS: _default_paths[path] = (root_path / DEFAULT_PATHS[path]).as_posix()
DEFAULT_PATHS = argparse.Namespace(**_default_paths)

if __name__ == "__main__":
    print(str(DEFAULT_PATHS)+"\n")
    print(str(MODEL_DEFAULTS)+"\n")
    print(str(DISCORD_BOT_SETTINGS)+"\n")
    print(str(CMD_SERVER_SETTINGS)+"\n")