Sept 19/2022 Update - Please do not use this, this is a temporary repository for breaking changes until they can be merged.

Installation:
 - clone / download this repository to a folder of your choice
 - download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 - open a conda prompt, then navigate to the folder where you cloned this repository
 - conda env create -f environment.yaml
 - edit g_diffuser_bot_config.py and g_diffuser_bot_defaults.py as appropriate, make sure to save!
 
 Running:
 - open a conda prompt, then navigate to the g-diffuser-bot folder
 - conda activate g_diffuser_bot
 - run the discord bot with python g_diffuser_bot.py
 - alternatively, run the CLI interface with python g_diffuser_cli.py
 - You can use the CLI interface interactively with python g_diffuser_cli.py --interactive

Updating:
 - Simply download and replace your files with those from those repository. You probably don't need to replace your config and default settings files.
