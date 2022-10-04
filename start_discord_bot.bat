@echo off
pushd %0\..\
cmd /k "conda activate g_diffuser & python g_diffuser_bot.py"
pause