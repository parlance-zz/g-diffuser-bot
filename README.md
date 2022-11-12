![https://www.stablecabal.org](https://raw.githubusercontent.com/parlance-zz/g-diffuser-bot/dev/part_of_stable_cabal.png) https://www.stablecabal.org

##  g-diffuser-bot - Discord bot and Interactive CLI for stable-diffusion-grpcserver

Nov 11-2022 Update: I've created a website to showcase a demo gallery of out-painting images made using g-diffuser bot - https://www.g-diffuser.com/

Nov 08-2022 Update: In/out-painting and img2img (aka "riffing") has (finally) been added to the Discord bot. New Discord bot command 'expand' allows you to change the canvas size of an input image while filling it with transparency, perfect for setting up out-painting.

Nov 07-2022 Update: This update adds support for **clip guided models** and new parameters to control them. For now clip guidance has a heavy performance penalty, but this will improve with optimization. This update also adds **negative prompt support** to both the CLI and discord bot, and changes the default loaded models to include SD1.5 and SD1.5 with (small) clip. This update also adds several **new samplers (dpmspp_1, dpmspp_2, dpmspp_3)**.

Nov 02-2022 Update: In/out-painting bugs are now fixed, and the interactive CLI now prints proper error messages.

Oct 22-2022 Update: Most of the annoying bugs in the Discord bot have now been fixed. The gRPC server now uses a Docker image which includes xformers support (large speed boost) as well as support for **stable-diffusion-v1-5 (which is the new default)**. The install instructions have changed accordingly, please find the changes below:

## Vision for the g-diffuser-bot project:
 - ~~The goal of the project is to provide the best possible front-end, interface, and utilities for the diffusers library and to enable regular users to access these powerful abilities with a free and easy-to-use package that supports their local GPU and as many OS's / platforms as possible. A core focus of the library will be on multi-modality generation tasks and other media types such as music.~~
 - Nov 01-2022: There are now many mature and rapidly evolving easy-to-use frontends and discord bots for Stable Diffusion. As a lone developer I do not have the time or resources to keep pace with these developments. I will do my best to continue to maintain the project, and may occasionally add new features, but this project is no longer my primary focus.

## Installation:
 - System Requirements: Windows 10 (1903+), Windows 11, or Linux (Ubuntu 20+), nvidia GPU with at least 8GB VRAM, ~10GB free space for model downloads
 - If you are running Windows 10 you may need to turn on "developer mode" before beginning the install instructions. Look for "developer settings" in the start menu.
 
 1)  git clone this repository to a folder of your choice (or click the green "code" button up top and click "download zip")
 2)  download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 3)  open a conda prompt (click on the start menu and look for "anaconda prompt"), then navigate to the folder where you cloned or downloaded this repository.
 4)  run "conda env create -f environment.yaml"
 5)  Install Docker Desktop:<br/>
         - On Windows: https://docs.docker.com/desktop/install/windows-install/ <br/>
         - On Linux: sudo apt install docker-desktop
 6)  Sign up for an account at https://huggingface.co/
 ```
 - Accept the terms of use for the models you wish to use:
(https://huggingface.co/runwayml/stable-diffusion-v1-5, https://huggingface.co/hakurei/waifu-diffusion-v1-4)

 - Go to https://huggingface.co/settings/tokens and create a new access token.

 - Open g_diffuser_config.py, find GRPC_SERVER_SETTINGS.hf_token = "YOUR_HUGGINGFACE_ACCESS_TOKEN_HERE"
   Replace the placeholder text with the access token you generated above and save the file.

 - If you're using the Discord bot this is also where you enter your Discord bot token and guild id.
   (https://discordapp.com/developers/applications/, https://www.writebots.com/discord-bot-token/)
```
Optional: edit g_diffuser_config.py and g_diffuser_defaults.py to change any other default settings of interest
 
## Running:
 1)  Run the discord bot by using "start_discord_bot.bat"
 2)  Run the interactive CLI by using "start_interactive_cli.bat"
 3)  You can (optionally) start the sdgrpcserver with "start_server.bat" to see the server console in it's own separate window

## Updating:
 - Simply git pull or download and replace your files with those from this repository. You probably won't need to replace your g_diffuser_config.py or g_diffuser_defaults.py files, but you may need to merge changes.

## Troubleshooting:
 - Better install instructions are (always) coming
 - Docker Desktop has a helpful GUI that you can use to keep track of the gRPC server and it's Docker "container". You can view the server parameters it was launched with, restart it or shut it down, and view the console output to track down any errors from the grpc server side. In some rare cases (or when updating) you may need to delete the existing "image" from the docker images list.
 - If you have questions or problems running g-diffuser-bot, please post as much detailed information as you can in (https://github.com/parlance-zz/g-diffuser-bot/discussions/categories/q-a), either myself or someone in the community may be able to help you. Thank you for your patience.