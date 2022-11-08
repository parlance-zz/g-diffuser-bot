from pathlib import Path
import argparse

DEFAULT_PATHS = argparse.Namespace()
MODEL_DEFAULTS = argparse.Namespace()
DISCORD_BOT_SETTINGS = argparse.Namespace()
GRPC_SERVER_SETTINGS = argparse.Namespace()
CLI_SETTINGS = argparse.Namespace()

# ******************** SETTINGS BEGIN ************************

CLI_SETTINGS.disable_progress_bars = True
# ****** todo *******
#CLI_SETTINGS.image_viewer_path = "C:\Program Files\IrfanView\i_view64.exe"
#CLI_SETTINGS.image_viewer_options = "/one /silent"

# IMPORTANT - Change the default paths here if desired, these are relative to the root path
DEFAULT_PATHS.models = "./models"
DEFAULT_PATHS.inputs = "./inputs"
DEFAULT_PATHS.outputs = "./outputs"
DEFAULT_PATHS.backups = "./backups"
DEFAULT_PATHS.saved = "./saved"
DEFAULT_PATHS.temp = "./temp"
DEFAULT_PATHS.debug = "./debug"
DEFAULT_PATHS.extensions = "./extensions"
DEFAULT_PATHS.bot = "./bot"
DEFAULT_PATHS.sdgrpcserver_docker_volume = ""

#IMPORTANT - If you want to use the discord bot, use enter your access token below:
# - Discord developers site: https://discordapp.com/developers/applications/
# - Discord bot setup guide: https://www.writebots.com/discord-bot-token/
DISCORD_BOT_SETTINGS.token = "YOUR_DISCORD_BOT_TOKEN_HERE"
DISCORD_BOT_SETTINGS.guilds = [1021168973582184548] # IMPORTANT - Enter your discord guild/server id(s) here
DISCORD_BOT_SETTINGS.model_list = ["stable-diffusion-v1-5", "stable-diffusion-v1-5-clip-small"] # IMPORTANT - model id's to be accessible from the discord bot (from ./models/models.yaml)
DISCORD_BOT_SETTINGS.max_queue_length = 1000     # beyond this limit additional commands will be rejected
DISCORD_BOT_SETTINGS.default_output_n = 1        # default batch size to create images
DISCORD_BOT_SETTINGS.max_output_limit = 3        # max number of samples to create simultaneously with -n param
DISCORD_BOT_SETTINGS.max_steps_limit = 120       # max number of steps per sample command
DISCORD_BOT_SETTINGS.accepted_attachments = [".png", ".jpg", ".jpeg"] # attachments in bot commands not matching this list will not be downloaded
DISCORD_BOT_SETTINGS.state_file_path = "./g-diffuser-bot.json"        # relative to root path
DISCORD_BOT_SETTINGS.activity = "/help, /about"

#IMPORTANT - GRPC server settings (you probably won't need to adjust these settings unless you are an advanced user)
GRPC_SERVER_SETTINGS.enable_local_network_access = True
GRPC_SERVER_SETTINGS.memory_optimization_level = 2  # 3 is maximum memory savings, 2 is normal, 1 is less, and 0 is off
                                                    # 2 should be the best setting for most users
GRPC_SERVER_SETTINGS.enable_mps = False
GRPC_SERVER_SETTINGS.nsfw_behaviour="ignore" # can be "flag", "block", or "ignore"
#GRPC_SERVER_SETTINGS.docker_image_name = "hafriedlander/stable-diffusion-grpcserver:xformers-latest"
#GRPC_SERVER_SETTINGS.docker_image_name = "hafriedlander/stable-diffusion-grpcserver:cuda118-xformers-0.0.2"
GRPC_SERVER_SETTINGS.docker_image_name = "parlance/sdgrpcserver-dev"

#IMPORTANT - Enter your hugging-face access token here to download models automatically as needed (https://huggingface.co/settings/tokens)
# - If you choose not to use a hugging-face access token you will need to place pre-downloaded (diffusers) models in the ./models folder; Please see ./models/README.md for more info.
GRPC_SERVER_SETTINGS.hf_token = "YOUR_HUGGINGFACE_ACCESS_TOKEN_HERE"
GRPC_SERVER_SETTINGS.refresh_models = False  # if enabled, this will check hugging-face for and download any changes to models that may already be downloaded / cached


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
    print("\ngrpc server settings: ")
    print_namespace(GRPC_SERVER_SETTINGS, debug=True)
    print("\ncli settings: ")
    print_namespace(CLI_SETTINGS, debug=True)