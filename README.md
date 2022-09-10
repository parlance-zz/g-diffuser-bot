Real instructions and documentation to come later, but if you're feeling brave:

Install instructions:
 - Install Python 3.8.5ish, in a conda env if you wish
 - pip install numpy
 - pip install image
 - pip scikit-image
 - pip install discord
 - pip install psutil
 - pip install pytimeparse
 - pip install torch
 - pip install --upgrade diffusers

You'll need a huggingface token if you don't already have one (or alternatively download the local model for the SD1.4 for diffusers), then edit g_diffuser_bot_params.py and review / change the values there. (https://huggingface.co/CompVis/stable-diffusion-v1-4)

If running python g_diffuser_bot.py errors out with an import error you can try using pip install whatever to install it. Email me at parlance@fifth-harmonic and I can probably help you out.

Good luck!

- The old t2i and i2i commands are gone, Instead use !gen to do everything now.
- If you don't attach an image when using !gen it will be treated as text to image.
- If you do attach an image but it has no alpha channel, it will be treated as image to image.
- If you do attach an image and it has an alpha channel, it will be used for in-painting.
- Parameter names are now: 
["-str", "-scale", "-seed", "-steps", "-x", "-mine", "-all", "-num", "-force", "-user", "-w", "-h", "-n", "-none", "-noise", "-noise_q"]
