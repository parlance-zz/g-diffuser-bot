from pathlib import Path
import argparse

DEFAULT_PATHS = argparse.Namespace()
MODEL_DEFAULTS = argparse.Namespace()
DISCORD_BOT_SETTINGS = argparse.Namespace()
GRPC_SERVER_SETTINGS = argparse.Namespace()

# ******************** SETTINGS BEGIN ************************

# IMPORTANT - Change the default paths here if desired, these are relative to the root path
DEFAULT_PATHS.models = "./models"
DEFAULT_PATHS.inputs = "./inputs"
DEFAULT_PATHS.outputs = "./outputs"
DEFAULT_PATHS.backups = "./backups"
DEFAULT_PATHS.temp = "./temp"
DEFAULT_PATHS.debug = "./debug"
DEFAULT_PATHS.extensions = "./extensions"
DEFAULT_PATHS.model_cfg = "./g_diffuser_config_models.yaml"
DEFAULT_PATHS.grpc_log = "./sdgrpcserver.log"

#IMPORTANT - If you want to use the discord bot, use enter your access token below:
# - Discord developers site: https://discordapp.com/developers/applications/
# - Discord bot setup guide: https://www.writebots.com/discord-bot-token/
DISCORD_BOT_SETTINGS.token = "YOUR_DISCORD_BOT_TOKEN_HERE"
DISCORD_BOT_SETTINGS.admin_role = "moderator" # IMPORTANT - use your discord server / guild role names here accordingly, check the bot help text for associated command permissions
DISCORD_BOT_SETTINGS.users_role = "everyone"  # if you want anyone to be able to use the bot, set this to "everyone"
DISCORD_BOT_SETTINGS.cmd_prefix = "!"               # all discord bot commands will require this prefix (e.g. !gen)
DISCORD_BOT_SETTINGS.queue_mode = 0                 # 0 for round-robin, 1 for first-come first-serve
DISCORD_BOT_SETTINGS.queue_poll_interval = 0.25     # how often should we look for new commands to begin processing (in seconds)
DISCORD_BOT_SETTINGS.max_queue_length = 1000        # beyond this limit additional commands will be rejected
DISCORD_BOT_SETTINGS.max_queue_print_items = 5      # max number of items to show for !queue command (up to discord message length limit)
DISCORD_BOT_SETTINGS.max_repeat_limit = 100         # max number of repetitions that can be used with the -x param
DISCORD_BOT_SETTINGS.max_output_limit = 3           # max number of samples to create simultaneously with -n param
DISCORD_BOT_SETTINGS.max_steps_limit = 100          # max number of steps per sample command
DISCORD_BOT_SETTINGS.default_n = 3                  # default number of samples per command
DISCORD_BOT_SETTINGS.accepted_attachments = [".png", ".jpg", ".jpeg"] # attachments in bot commands not matching this list will not be downloaded
DISCORD_BOT_SETTINGS.state_file_path = "./g-diffuser-bot.json"        # relative to root path
DISCORD_BOT_SETTINGS.activity = DISCORD_BOT_SETTINGS.cmd_prefix + "help, " + DISCORD_BOT_SETTINGS.cmd_prefix + "about"

#IMPORTANT - GRPC server settings (you probably won't need to adjust these settings unless you are an advanced user)
GRPC_SERVER_SETTINGS.host = "localhost:50051"
GRPC_SERVER_SETTINGS.key = ""
GRPC_SERVER_SETTINGS.memory_optimization_level = 1  # 2 is maximum memory savings, 1 is less, and 0 is off
                                                    # saving more memory generally means mildly decreased performance or output coherence
GRPC_SERVER_SETTINGS.enable_mps = False

# ******************** SETTINGS END ************************

# make global default paths namespace
root_path = Path(__file__).parent.absolute()
_default_paths = {"root": root_path.as_posix()}
DEFAULT_PATHS = vars(DEFAULT_PATHS)
for path in DEFAULT_PATHS: _default_paths[path] = (root_path / DEFAULT_PATHS[path]).as_posix()
DEFAULT_PATHS = argparse.Namespace(**_default_paths)

if __name__ == "__main__": # you can execute this file with python to see a summary of your config
    from g_diffuser_lib import print_namespace
    print("\ndefault paths: ")
    print_namespace(DEFAULT_PATHS, debug=True)
    print("\ndiscord bot settings: ")
    print_namespace(DISCORD_BOT_SETTINGS, debug=True)
    print("\ngrpc server settings: ")
    print_namespace(GRPC_SERVER_SETTINGS, debug=True)