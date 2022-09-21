# --- bot and huggingface private tokens -----

# IMPORTANT - Sign up for a Discord developer account if you don't have one, create a new app,
# create a bot, then put your Discord bot token here
# Discord developers site: https://discordapp.com/developers/applications/
# Discord bot setup guide: https://www.writebots.com/discord-bot-token/
BOT_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"

# IMPORTANT - use your server / guild role names here accordingly, check the help text for associated command permissions
BOT_ADMIN_ROLE_NAME = "YOUR_ADMIN_ROLE_NAME_HERE"  
BOT_USERS_ROLE_NAME = "YOUR_USER_ROLE_NAME_HERE" # if you want anyone to be able to use the bot, change this to "everyone"
BOT_COMMAND_PREFIX = "!"

# IMPORTANT - Sign up for an account on https://www.huggingface.co and accept the terms to use stable diffusion
# Create an access token at https://huggingface.co/settings/tokens and enter your token below:
#HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_ACCESS_TOKEN_HERE"
#CMD_SERVER_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
# IMPORTANT - If you use the local model download instead for diffusers stable-diffusion use these lines
HUGGINGFACE_TOKEN = None
CMD_SERVER_MODEL_NAME = "./stable-diffusion-v1-4"  # local relative path to downloaded model

DEBUG_MODE = False # if you want more detailed error information and debug file dumps enable this

# default paths - change if you need to
ROOT_PATH = Path(__file__).parent.absolute()
TMP_ROOT_PATH = (ROOT_PATH / "tmp").as_posix()                       # make sure this is a valid path (it will be created if it does not exist)
BACKUP_PATH = (ROOT_PATH / "backup").as_posix()                      # make sure this is a valid path (it will be created if it does not exist)
BOT_STATE_DATA_FILE = (ROOT_PATH / "g-diffuser-bot.json").as_posix() # used for persisting command queue, top user list and input image paths
ROOT_PATH = ROOT_PATH.as_posix()
