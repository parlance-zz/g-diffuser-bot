Real instructions and documentation to come later, but if you're feeling brave:

To install it you don't need any kind of conda env, just install python, then install diffusers with pip install --upgrade git+https://github.com/huggingface/diffusers

You'll need a huggingface token if you don't already have one (or alternatively download the local model for the SD1.4 for diffusers), then edit the params file as usual.

If running python g_diffuser_bot.py errors out with an import error you can try using pip install whatever to install it.

Good luck!

- The old t2i and i2i commands are gone, Instead use !gen to do everything now.
- If you don't attach an image when using !gen it will be treated as text to image.
- If you do attach an image but it has no alpha channel, it will be treated as image to image.
- If you do attach an image and it has an alpha channel, it will be used for in-painting.
- WARNING: Be careful when erasing images for in-painting. The windows clipboard will destroy color values in transparent pixels, and some editing programs
           need specific export options to avoid destroying those color values. If your in-painting looks funky or has a huge blotch where you erased, then
           this is probably what happened.
- Parameter names are now: ["-str", "-scale", "-seed", "-steps", "-x", "-mine", "-all", "-num", "-force", "-user", "-w", "-h", "-n", "-none"]
-          !gen this is an example prompt -str 0.5 -scale 12 -n 6
