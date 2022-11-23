from pathlib import Path
import argparse

DEFAULT_PATHS = argparse.Namespace()
MODEL_DEFAULTS = argparse.Namespace()
DISCORD_BOT_SETTINGS = argparse.Namespace()
GRPC_SERVER_SETTINGS = argparse.Namespace()
CLI_SETTINGS = argparse.Namespace()

# ******************** SETTINGS BEGIN ************************

CLI_SETTINGS.disable_progress_bars = True

# IMPORTANT - Change the default paths here if desired, these are relative to the root path
DEFAULT_PATHS.models = "../models"
DEFAULT_PATHS.inputs = "./inputs"
DEFAULT_PATHS.outputs = "./outputs"
DEFAULT_PATHS.temp = "./temp"
DEFAULT_PATHS.bot = "./bot"

DISCORD_BOT_SETTINGS.guilds = [1021168973582184548] # IMPORTANT - Enter your discord guild/server id(s) here so the bot will join them
DISCORD_BOT_SETTINGS.model_list = ["stable-diffusion-v1-5-clip-small"] # IMPORTANT - model id's to be accessible from the discord bot (from ./models/models.yaml)
DISCORD_BOT_SETTINGS.max_queue_length = 1000     # beyond this limit additional commands will be rejected
DISCORD_BOT_SETTINGS.default_output_n = 1        # default batch size to create images
DISCORD_BOT_SETTINGS.max_output_limit = 3        # max number of samples to create simultaneously with -n param
DISCORD_BOT_SETTINGS.max_steps_limit = 100       # max number of steps per sample command
DISCORD_BOT_SETTINGS.accepted_attachments = ["image/png", "image/jpg", "image/jpeg"] # attachments in bot commands not matching this list will not be downloaded
DISCORD_BOT_SETTINGS.state_file_path = "./g_diffuser_bot.json"        # relative to root path
DISCORD_BOT_SETTINGS.activity = ""

# ******************** SETTINGS END ************************

# make global default paths namespace
root_path = Path(__file__).parent.absolute()
_default_paths = {"root": root_path.as_posix()}
DEFAULT_PATHS = vars(DEFAULT_PATHS)
for path in DEFAULT_PATHS: _default_paths[path] = (root_path / DEFAULT_PATHS[path]).as_posix() if DEFAULT_PATHS[path] != "" else ""
DEFAULT_PATHS = argparse.Namespace(**_default_paths)

if __name__ == "__main__": # you can execute this file with python to see a summary of your config
    from extensions.g_diffuser_lib import print_namespace
    print("\ndefault paths: ")
    print_namespace(DEFAULT_PATHS, debug=True)
    print("\ndiscord bot settings: ")
    print_namespace(DISCORD_BOT_SETTINGS, debug=True)
    print("\ncli settings: ")
    print_namespace(CLI_SETTINGS, debug=True)