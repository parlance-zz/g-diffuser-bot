# zoom and  E N H A N C E

import os
import shutil
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

# put your prompt for sampling here
my_prompt = 'a man in a hoodie standing in front of a bunch of plants, cyberpunk art, by Android Jones, cgsociety, analytical art, an anthropomorphic stomach, protagonist in foreground, logo without text, the head of the man is a skull, dark neighborhood, art for the game, bleed, in the anime series ergo proxy'
init_img = "endzoom.png" # starting (or rather, ending image)
num_steps = 100          # number of discrete zoom images to sample
expand_top = 40.         # amount to expand in each direction in each step
expand_bottom = 40.
expand_left = 40.
expand_right = 40.
expand_softness = 30.
expand_space = 1.

args = gdl.get_default_args() # sampling params
args.prompt = my_prompt
args.init_img = init_img
args.steps = 50
args.scale = 12.
args.sampler = "k_euler_ancestral"

expanded_img = os.path.splitext(init_img)[0]+".expanded.png"

# create zoom frames
for i in range(num_steps):
    print("Starting iteration {0} of {1}...".format(i+1, num_steps))
    expand(args.init_img, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space, output_file=expanded_img)
    args.init_img = expanded_img
    args = sample(**vars(args))

    output_file = DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = DEFAULT_PATHS.inputs+"/"+args.init_img
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

print("Done!")