# zoom and  E N H A N C E

import os
import shutil
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

# put your prompt for sampling here
my_prompt = "art by remedios varo"
init_img = "endzoom.png"  # starting (or rather, ending image)
num_frames = 100000       # number of discrete zoom images to sample
expand_top = 50           # amount to expand in each direction in each step
expand_bottom = 50        # these values are in % of the original image size
expand_left = 50
expand_right = 50
expand_softness = 85.
expand_space = 1.

args = gdl.get_default_args() # sampling params
args.prompt = my_prompt
args.init_img = init_img
args.steps = 100
args.scale = 8.
args.guidance_scale = 0.25
args.noise_start = 1.618
#args.guidance_scale = 0.
#args.sampler = "k_euler_ancestral"

expanded_img = os.path.splitext(init_img)[0]+".expanded.png"

# create zoom frames
for i in range(num_frames):
    print("Starting iteration {0} of {1}...".format(i+1, num_frames))
    expand(args.init_img, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space, output_file=expanded_img)
    args.init_img = expanded_img
    args = sample(**vars(args))

    output_file = DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = DEFAULT_PATHS.inputs+"/"+args.init_img
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

print("Done!")