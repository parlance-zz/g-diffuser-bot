![https://www.stablecabal.org](https://raw.githubusercontent.com/parlance-zz/g-diffuser-bot/dev/part_of_stable_cabal.png) https://www.stablecabal.org

##  g-diffuser-bot - Discord bot and Interactive CLI for stable-diffusion-grpcserver

Oct 22-2022 Update: Most of the annoying bugs in the Discord bot have now been fixed. The gRPC server now uses a Docker image which includes xformers support (large speed boost) as well as support for **stable-diffusion-v1-5 (which is the new default)**. The install instructions have changed accordingly, please find the changes below:

## Vision for the g-diffuser-bot project:
 - ~~The goal of the project is to provide the best possible front-end, interface, and utilities for the diffusers library and to enable regular users to access these powerful abilities with a free and easy-to-use package that supports their local GPU and as many OS's / platforms as possible. A core focus of the library will be on multi-modality generation tasks and other media types such as music.~~
 - Oct 11-2022: There are now many mature and rapidly evolving easy-to-use frontends and discord bots for Stable Diffusion. As a lone developer I do not have the time or resources to keep pace with these developments. I will do my best to continue to maintain the project, and may occasionally add new features, but this project is no longer my primary focus.

## Installation:
 - If you are running Windows 10 you may need to turn on "developer mode". Look for "developer settings" in the start menu.
 
 1)  git clone this repository to a folder of your choice (or click the green "code" button up top and click "download zip")
 2)  download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 3)  open a conda prompt (click on the start menu and look for "anaconda prompt"), then navigate to the folder where you cloned or downloaded this repository.
 4)  run "conda env create -f environment.yaml"
 5)  Install Docker Desktop:
         On Windows: https://docs.docker.com/desktop/install/windows-install/
         On Linux: sudo apt install docker-desktop
 6)  Sign up for an account at https://huggingface.co/

         - Accept the terms of use for the models that you wish to use (https://huggingface.co/CompVis/stable-diffusion-v1-4, https://huggingface.co/hakurei/waifu-diffusion)

         - Go to https://huggingface.co/settings/tokens and create a new access token.

         - Open g_diffuser_config.py and find line 54 (GRPC_SERVER_SETTINGS.hf_token = "YOUR_HUGGINGFACE_ACCESS_TOKEN_HERE"); replace this placeholder text with the access token you just generated and save the file.
         
         - If you wish to use the Discord bot, this file is also where you should enter your Discord bot token and guild id.

Optional: edit g_diffuser_config.py and g_diffuser_defaults.py to change default settings
 
## Running:
 1)  open a conda prompt (click on the start menu and look for "anaconda prompt"), then navigate to the g-diffuser folder
 2)  run "conda activate g_diffuser" (OPTIONAL: on Windows you can open start_prompt.bat to do these 2 steps automatically)
 3)  run the discord bot with: "python g_diffuser_bot.py"
       - alternatively, run the CLI interface with: "python g_diffuser_cli.py"
       - You can use the CLI interface interactively with: "python g_diffuser_cli.py --interactive"
       - On Windows you can open start_interactive_cli.bat to open the interactive cli directly in one step instead of the above

## Updating:
 - Simply git pull or download and replace your files with those from this repository. You probably won't need to replace your g_diffuser_config.py or g_diffuser_defaults.py files, but you may need to merge changes.

## Troubleshooting:
 - Better install instructions are (always) coming
 - Docker Desktop has a helpful GUI that you can use to keep track of the gRPC server and it's Docker "container". You can view the server parameters it was launched with, restart it or shut it down, and view the console output to track down any errors from the grpc server side.
 - If you have questions or problems running anything in g-diffuser-bot, please post as much detailed information as you can in (https://github.com/parlance-zz/g-diffuser-bot/discussions/categories/q-a), myself or someone in the community may be able to help you. Thank you for your patience.
