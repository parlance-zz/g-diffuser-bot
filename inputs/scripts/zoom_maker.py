# zoom and  E N H A N C E

import os
import shutil
import cv2
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

# put your prompt for sampling here
my_prompt = "Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed fantasy, rich colors, high contrast, gloomy atmosphere, dark background. trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt" #"art by studio ghibli"
#my_prompt = "rectilinear lens"

init_img = "endzoom.png"  # starting (or rather, ending image)
num_frames = 100000       # number of discrete zoom images to sample

expand_softness = 100.
expand_space = 1.
expand_top = 30           # amount to expand in each direction in each step
expand_bottom = 30        # these values are in % of the original image size
expand_left = 30          # exceeding 50% in any direction is not recommended for recursive zooms / pans
expand_right = 30

args = gdl.get_default_args() # sampling params
args.prompt = my_prompt
args.init_img = init_img
args.output_path = "zoom_maker"
args.output_name = "zoom_maker"
args.steps = 100  #100
args.scale = 14. #14. #8.
args.guidance_strength = 1. #0.4 #0.3 #0.25  # try lowering clip guidance_strength if you have problems with zooms "exploding"
args.noise_start = 2.
args.negative_prompt = "frame, panel, comic book, collage, cropped, saturated, watermark, text, logo, signature, greyscale, monotone, vignette"

#args.sampler = "k_euler_ancestral"

# *****************************************************************

expanded_img = os.path.splitext(init_img)[0]+".expanded.png"
cv2_image = cv2.imread(DEFAULT_PATHS.inputs+"/"+init_img)
original_size = cv2_image.shape

# create zoom frames
for i in range(num_frames):
    print("Starting iteration {0} of {1}...".format(i+1, num_frames))
    expand(args.init_img, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space, output_file=expanded_img, add_noise=1.)
    args.init_img = expanded_img
    args = sample(**vars(args))

    output_file = DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = DEFAULT_PATHS.inputs+"/"+args.init_img
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

print("Done!")