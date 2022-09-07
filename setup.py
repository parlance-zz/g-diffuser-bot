# discord, psutil, and urlparse are the only modules you need outside of the pre-built ldm conda environment

try: # install discord module if we haven't already
    import discord
except ImportError:
    print("Could not import discord, installing with pip...")
    from pip._internal import main as pip
    pip(['install', 'discord'])
    
# psutil is used to reliably force-close commands that have been cancelled while running
try: # install psutil module if we haven't already
    import psutil
except ImportError:
    print("Could not import psutil, installing with pip...")
    from pip._internal import main as pip
    pip(['install', 'psutil'])
    
# pytimeparse

# install diffusers
# pip install --upgrade git+https://github.com/huggingface/diffusers

"""
# offline model download for diffusers
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
"""