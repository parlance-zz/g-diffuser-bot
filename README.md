Copyright (c) 2022 Christopher Friesen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Sept 19/2022 Update - Please do not use this, this is a temporary repository for breaking changes until they can be merged.


Installation:
 - clone / download this repository to a folder of your choice
 - download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 - open a conda prompt, then navigate to the folder where you cloned / downloaded this repository
 - conda env create -f environment.yaml
 - edit g_diffuser_bot_config.py and g_diffuser_bot_defaults.py as appropriate, make sure to save!
 
 Running:
 - open a conda prompt, then navigate to the g-diffuser-bot folder
 - conda activate g_diffuser_bot
 - run the discord bot with python g_diffuser_bot.py
 - alternatively, run the CLI interface with python g_diffuser_cli.py
 - You can use the CLI interface interactively with python g_diffuser_cli.py --interactive
 - If you see an out of memory error set BOT_USE_OPTIMIZED = True in g_diffuser_bot_defaults.py, or use --use_optimized when running g_diffuser_cli.py

Updating:
 - Simply download and replace your files with those from this repository. You probably won't need to replace your config and default settings files.
