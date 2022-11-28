# zoom and  E N H A N C E

import os
import shutil
import cv2
import numpy as np

from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

# put your prompt for sampling here
my_prompts = [
   #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed fantasy, trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
   #"a wooden shed, many green plant and flower gowing on it, illustration key visual trending pixiv fanbox by wlop and greg rutkowski and makoto shinkai and studio ghibli",
   #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed fantasy, rich colors, high contrast, gloomy atmosphere, dark background. trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
   #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed science fiction, rich colors, high contrast, gloomy atmosphere, dark background. trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
   "entrance of the huge castle surrounded by advanced brutalist architecture, art by moebius, irakli nadar, overdetailed art, colorful, artistic record jacket design",
]
prompt_reset_interval = 2   # the prompt is switched to the next prompt in the list every n samples
prompt_schedule = "linear"  # rotate through the prompt list in order
#prompt_schedule = "random" # uncomment this line to use prompts in random order

init_img = "endzoom.png"  # starting (or rather, ending image)
num_frames = 100000       # number of discrete zoom images to sample
                          # (you can abort / close the program at any time to use the keyframes you have already generated)

expand_softness = 100.
expand_space = 1.
expand_top = 30           # amount to expand in each direction in each step
expand_bottom = 30        # these values are in % of the original image size
expand_left = 30          # exceeding 50% in any direction is not recommended for recursive zooms / pans
expand_right = 30

args = gdl.get_default_args() # sampling params
args.init_img = init_img
args.output_path = "zoom_maker"
args.output_name = "zoom_maker"
args.steps = 60 #12 
args.scale = 14.#14. 
args.guidance_strength = 0.5   # try lowering clip guidance_strength if you have problems with zooms "exploding"
args.noise_start = 2.
#args.negative_prompt = "frame, comic book, collage, cropped, oversaturated, signed, greyscale, monotone, vignette, title, text, logo, watermark"
args.negative_prompt = "watermark, title, label, logo, collage, cropped, oversaturated, monotone, vignette"

#args.sampler = "k_euler_ancestral"
#args.sampler = "dpmspp_2"
args.sampler = "dpmspp_3"

# *****************************************************************

expanded_img = os.path.splitext(init_img)[0]+".expanded.png"
cv2_image = cv2.imread(DEFAULT_PATHS.inputs+"/"+init_img)
original_size = cv2_image.shape

# create zoom frames
for i in range(num_frames):
    print("Starting iteration {0} of {1}...".format(i+1, num_frames))
    expand(args.init_img, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space, output_file=expanded_img)

    if ((i % prompt_reset_interval) == 0) or (i == 0):
        if prompt_schedule == "linear":
            prompt_index = (i // prompt_reset_interval) % len(my_prompts)
        elif prompt_schedule == "random":
            prompt_index = np.random.randint(0, len(my_prompts))
        else:
            raise Exception("Unknown prompt schedule '{0}'".format(prompt_schedule))
        args.prompt = my_prompts[prompt_index]
        print("prompt: {0}".format(args.prompt))

    args.init_img = expanded_img
    args = sample(**vars(args))

    output_file = DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = DEFAULT_PATHS.inputs+"/"+args.init_img
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

print("Done!")