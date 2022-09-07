# --- bot params ----------------------------------------------------------------------------

# IMPORTANT - Sign up for a Discord developer account if you don't have one, create a new app, bot, then put your Discord bot token here
# Discord developers site: https://discordapp.com/developers/applications/ instructions: https://www.writebots.com/discord-bot-token/
BOT_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"

BOT_ADMIN_ROLE_NAME = "YOUR_ADMIN_ROLE_NAME_HERE"  # IMPORTANT - use your server / guild group names here accordingly, check the help text for command permissions
BOT_USERS_ROLE_NAME = "YOUR_USER_ROLE_NAME_HERE" # if you want anyone to be able to use the bot, change this to "everyone"

BOT_COMMAND_PREFIX = "!"
BOT_ACTIVITY = "The way of the future..."
BOT_USE_OPTIMIZED = False             # IMPORTANT - Set to true if you encounter errors running commands due to low memory

# IMPORTANT - Sign up for an account on https://www.huggingface.co and accept the terms to use stable diffusion
# Create an access token at https://huggingface.co/settings/tokens and enter your token below:
#CMD_SERVER_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
CMD_SERVER_MODEL_SCHEDULER = "default" # samplers are called schedulers in the diffusers library
#HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE" 

# alternatively if you follow the local download instructions, make this the local relative path to the downloaded folder"
CMD_SERVER_MODEL_NAME = "./stable-diffusion-v1-4" # local folder download path instead of token
#CMD_SERVER_MODEL_SCHEDULER = "LMS" # this is the only alternative "sampler" in the diffusers library right now
HUGGINGFACE_TOKEN = None
                       
# default paths
from pathlib import Path
ROOT_PATH = str(Path(__file__).parent.absolute())
TMP_ROOT_PATH = ROOT_PATH + "/tmp"                               # make sure this is a valid path (it will be created if it does not exist)
BACKUP_PATH = ROOT_PATH + "/backup"                              # make sure this is a valid path (it will be created if it does not exist)
ESRGAN_PATH = ROOT_PATH + "/esrgan/realesrgan-ncnn-vulkan.exe"   # optional, but required for !enhance
BOT_STATE_DATA_FILE = ROOT_PATH + "/g-diffuser-bot.json"       # used for persisting command queue, top user list and input image paths
BOT_LOG_FILE_PATH = ROOT_PATH + "/g-diffuser-bot.log"          # keep a text log of commands

# default params for commands, these are overriden by any user supplied params
DEFAULT_CMD_PARAMS = { "-n": 1, "-steps": "50", "-scale": "10", "-str": "0.42", "-w": "512", "-h": "512" }
ESRGAN_PARAMS = { "-n": "realesrgan-x4plus" }
AUTO_SEED_RANGE = (1,999999)
MAX_STEPS_LIMIT = 300
MAX_REPEAT_LIMIT = 100             # max number of repetitions that can be used with the -x param
MAX_OUTPUT_LIMIT = 6               # max number of samples to create with -n param
MAX_RESOLUTION = (768, 768)        # max resolution to avoid erroring out of memory
MAX_STRENGTH = 0.999999999

QUEUE_MODE = 0                     # 0 for round-robin, 1 for first come first serve
QUEUE_POLL_INTERVAL = 0.5          # how often should the queue look for new commands to begin processing (in seconds)
MAX_QUEUE_LENGTH = 1000            # beyond this limit additional commands will be rejected
MAX_QUEUE_PRINT_ITEMS = 4          # max number of items to show for !queue command (up to discord message length limit)

ACCEPTED_ATTACHMENTS = [".png", ".jpg", ".jpeg"] # attachments in bot commands not matching this list will not be downloaded

TMP_CLEAN_PATHS = [
    TMP_ROOT_PATH + "/*.*",
    BACKUP_PATH + "/*.json",
]

# merged list of all valid options / parameters for all commands
PARAM_LIST = ["-str", "-scale", "-seed", "-steps", "-x", "-mine", "-all", "-num", "-force", "-user", "-w", "-h", "-n", "-none"]

# by default the command server runs and binds to localhost for local connections only
CMD_SERVER_BIND_HTTP_HOST = "localhost"
CMD_SERVER_BIND_HTTP_PORT = 39132

# -------------------------------------------------------------------------------------------

# these strings must be 2000 characters or less

ABOUT_TXT = """This is a simple discord bot for stable-diffusion and provides access to the most common commands as well as a few others.

Commands can be used in any channel the bot is in, provided you have the appropriate server role. For a list of commands, use !help

Please use discretion in your prompts as the safety filter has been disabled. Repeated violations will result in banning.
If you do happen to generate anything questionable please delete the message yourself or contact a mod ASAP. The watermarking feature has been left enabled to minimize potential harm.

For more information on the G-Diffuser-Bot please see https://github.com/parlance-zz/g-diffuser-bot
"""

HELP_TXT1 = """
User Commands:
  !t2i : Generates an image with a prompt [-seed num] [-scale num] [-steps num][-x num]
  !t2ihd : As above, uses txt2imghd to generate 1 sample at 4x size
  !i2i : Generates an image with a prompt and input image [-seed num] [-str num] [-scale num] [-steps num] [-x num] 
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
  -x: Repeat the given command some number of times. Instead of a number, use -x samplers to repeat the command with every sampler.

Models and Samplers:
 - !t2i supports alternate samplers, and all generation commands support alternate models with [-m model] (sd1.4_small)
 - To use an alternate sampler use the following options [-plms] [-dpm_2_a] [-dpm_2] [-euler_a] [-euler] [-heun] [-lms]
 
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
!t2ihd baroque painting of a mystical island treehouse on the ocean, chrono trigger, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate -plms
!t2i my little pony in space marine armor from warhammer 40k, trending on artstation, intricate detail, 3d render, gritty, dark colors, cinematic lighting, cosmic background with colorful constellations -scale 10 -seed 174468 -steps 50
!t2ihd baroque painting of a mystical island treehouse on the ocean, chrono trigger, trending on artstation, soft lighting, vivid colors, extremely detailed, very intricate -scale 14 -str 0.375 -seed 252229
"""