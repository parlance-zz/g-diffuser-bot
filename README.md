Sept 11/2022 Update - The BUG-FREE-ish implementation of the in/out-paint system is finally complete. Please try it ASAP! It's awesome!!

With the Sept-11 update please use the following default params: -str 0.6, -scale 10, -blend 10, -color 0.005

Edit: I have a feeling the outpaint implementation may garner some traffic. Please forgive the implementation as I've been very busy and trying
to write code as fast as psosible. Thank you.

Real instructions and documentation to come later, but if you're feeling brave:

Install instructions:
 - Install conda and python 3.8.5ish
 - In an conda env of your choosing:
   - conda install -c pytorch torchvision cudatoolkit pytorch 
   - pip install numpy image scikit-image discord psutil pytimeparse diffusers transformers
 
You'll need a huggingface token if you don't already have one (or alternatively download the local model for the SD1.4 for diffusers), then edit g_diffuser_bot_params.py and review / change the values there. (https://huggingface.co/CompVis/stable-diffusion-v1-4)

If running python g_diffuser_bot.py errors out with an import error you can try using pip install whatever to install it. Email me at parlance@fifth-harmonic and I can probably help you out.

Good luck!

- Use !help to get a command and parameter overview
- If you don't attach an image when using !gen it will be treated as text to image.
- If you do attach an image but it has no alpha channel, it will be treated as image to image.
- If you do attach an image and it has an alpha channel, it will be used for in-painting.
- You don't need to worry about "erasing color values" under transparency, but when erasing try to avoid cutting things out into squares.
