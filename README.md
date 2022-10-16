![https://www.stablecabal.org](https://raw.githubusercontent.com/parlance-zz/g-diffuser-bot/dev/part_of_stable_cabal.png) https://www.stablecabal.org

##  g-diffuser-bot - Discord bot and utilities for the diffusers library (stable-diffusion)

Oct 9/2022 Update:  I'm recharging over thanksgiving and will working on the bot again starting Oct-11 2022.
                    Please check out the other GRPC server interfaces available on: https://www.stablecabal.org/
                    Thank you for your patience.

Oct 4/2022 Update: The discord bot is back! I'll be working to re-add features as I go, including gui buttons et al.

## Docker File
- docker image and other simplified installation options for g-diffuser-bot will be published very soon!

## Vision for the g-diffuser-bot project:
 - In the near future the diffusers library (https://github.com/huggingface/diffusers) being developed and promoted by stability.ai will expose multi-modality sampling abilities, meaning we will be able to arbitrarily mix and match input and output types. Tasks like txt2music, music2img, and everything in-between will be possible, and all of this will be coming very soon.
 - The goal of the project is to provide the best possible front-end, interface, and utilities for the diffusers library and to enable regular users to access these powerful abilities with a free and easy-to-use package that supports their local GPU and as many OS's / platforms as possible.
 - The current frontends include an (interactive) cli and a discord bot.
 - The current experimental extensions include g-diffuser fourier shaped noise out-painting
   - Fourier shaped noise has been integrated into hafriedlander's unified diffusers pipeline and development will continue in his GRPC server project: https://github.com/hafriedlander/stable-diffusion-grpcserver
 - For more information on progress and upcoming features please see: https://github.com/parlance-zz/g-diffuser-bot/issues
 
## Development and funding:
 - Donations are greatly appreciated and will be directly used to fund further development.
 - https://github.com/sponsors/parlance-zz
 - ETH to 0x8e4BbD53bfF9C0765eE1859C590A5334722F2086

## Installation:
 1)  clone this repository to a folder of your choice (or click the green "code" button up top and click "download zip")
     after this step you may need to also run "git submodule update --init --recursive" from the main folder.
 2)  download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 3)  open a conda prompt (click on the start menu and look for "anaconda prompt"),
     then navigate to the folder where you cloned / downloaded this repository.
 4)  run "conda env create -f environment.yaml"
 5)  place any pre-downloaded diffusers models into the ./models folder
     for specific instructions on model download / installation please see models/README.md (https://github.com/parlance-zz/g-diffuser-bot/tree/main/models)
 6)  If you are running Windows 10 you may need to turn on "developer mode". Look for "developer settings" in the start menu.
     
Optional: edit g_diffuser_config_models.yaml, g_diffuser_config.py and g_diffuser_defaults.py as appropriate, save your changes
 
## Running:
 1)  open a conda prompt (click on the start menu and look for "anaconda prompt"), then navigate to the g-diffuser folder
 2)  run "conda activate g_diffuser" (OPTIONAL: on Windows you can open start_prompt.bat to do these 2 steps automatically)
 3)  run the discord bot with: "python g_diffuser_bot.py"
       - alternatively, run the CLI interface with: "python g_diffuser_cli.py"
       - You can use the CLI interface interactively with: "python g_diffuser_cli.py --interactive"
       - On Windows you can open start_interactive_cli.bat to open the interactive cli directly in one step instead of the above
       - If you see an out of memory error check the settings and config files for low-memory options
       - Verify your configuration by running: "python g_diffuser_config.py" or: "python g_diffuser_defaults.py"

## Updating:
 - Simply git pull or download and replace your files with those from this repository. You probably won't need to replace your g_diffuser_config.py or g_diffuser_defaults.py files, but you may need to merge changes.

## Troubleshooting:
 - Better install instructions and an updated easy-to-isntall package are coming, but in the mean-time if you see any dependency errors you can try to fix them with "pip install xyz". Git really doesn't like to make downloading projects with submodules pain-free, so if you see errors related to the GRPC server please try "git submodule update --init --recursive".
 - If you have questions or problems running anything in g-diffuser-bot, please post as much detailed information as you can in (https://github.com/parlance-zz/g-diffuser-bot/discussions/categories/q-a), myself or someone in the community may be able to help you. Thank you for your patience.
 - Improved usage guides and setup instructions are coming soon.