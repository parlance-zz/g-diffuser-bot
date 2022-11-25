![https://www.stablecabal.org](https://www.g-diffuser.com/stablecabal.png) https://www.stablecabal.org

## [g-diffuser-bot](https://www.g-diffuser.com) - Discord bot and interface for Stable Diffusion
- [G-Diffuser / Stable Cabal Discord](https://discord.gg/stFy2UPppg)

Nov 23-2022 Update: The first release of the all-in-one installer version of G-Diffuser is here. This release no longer requires the installation of WSL or Docker, and has a systray icon to keep track of and launch G-Diffuser components. The download link is available under this project's [releases](https://github.com/parlance-zz/g-diffuser-bot/releases/tag/aio-release).

Nov 20-2022 Update: The infinite zoom scripts have been updated with some improvements, notably a new compositer script that is hundreds of times faster than before. The environment / requirements have changed, so if you want to use the new zoom compositer be sure to run a "conda env update -f environment.yaml"

Nov 19-2022 Update: There are some new g-diffuser CLI scripts that can be used to make infinite zoom videos. Check out [/inputs/scripts/](https://github.com/parlance-zz/g-diffuser-bot/tree/dev/inputs/scripts) and have a look at [zoom_maker](https://github.com/parlance-zz/g-diffuser-bot/blob/dev/inputs/scripts/zoom_maker.py) and [zoom_composite](https://github.com/parlance-zz/g-diffuser-bot/blob/dev/inputs/scripts/zoom_composite.py)

Nov 11-2022 Update: I've created a website to showcase a demo gallery of out-painting images made using g-diffuser bot - https://www.g-diffuser.com/

Nov 08-2022 Update: In/out-painting and img2img (aka "riffing") has (finally) been added to the Discord bot. New Discord bot command 'expand' allows you to change the canvas size of an input image while filling it with transparency, perfect for setting up out-painting.

Nov 07-2022 Update: This update adds support for **clip guided models** and new parameters to control them. For now clip guidance has a heavy performance penalty, but this will improve with optimization. This update also adds **negative prompt support** to both the CLI and discord bot, and changes the default loaded models to include SD1.5 and SD1.5 with (small) clip. This update also adds several **new samplers (dpmspp_1, dpmspp_2, dpmspp_3)**.

## System Requirements:
 - Windows 10 (1903+), Windows 11, or Linux (Ubuntu 20+), nvidia GPU with at least 8GB VRAM, ~40GB free space for model downloads
 - If you are running Windows 10/11 you may need to turn on "developer mode" before beginning the install instructions. Look for "developer settings" in the start menu.

## G-Diffuser all-in-one
The first release of the all-in-one installer is here. It notably features much easier "one-click" installation and updating, as well as a systray icon to keep track of g-diffuser programs and the server while it is running.

## Installation / Setup
- Run install_or_update.cmd at least once (once to install, and again later if you wish update to the latest version)
- Edit the filed named "config" and make sure to add your hugging-face access token and save the file.
  - If you don't have a huggingface token yet
    - Register for a HuggingFace account at https://huggingface.co/join
    - Follow the instructions to access the repository at https://huggingface.co/CompVis/stable-diffusion-v1-4 (don't worry, this doesn't mean SD1.4 will be downloaded or used, it just grants you the necessary access to download stable diffusion models)
    - Create a token at https://huggingface.co/settings/tokens

## Usage
- Run run.cmd to start the G-Diffuser system
- You should see a G-Diffuser icon in your systray / notification area. Click on the icon to open and interact with the G-Diffuser system. If the icon is missing be sure it isn't hidden by clicking the "up" arrow near the notification area.

![G-Diffuser Systray](https://www.g-diffuser.com/systray_screenshot.jpg)

GUI is coming soon(tm)