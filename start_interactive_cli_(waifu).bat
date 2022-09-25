@echo off
pushd %0\..\
REM todo: maybe check if the waifu-diffusion model is downloaded and display error if it isn't
cmd /k "conda activate g_diffuser & python g_diffuser_cli.py --interactive --model-name waifu-diffusion"