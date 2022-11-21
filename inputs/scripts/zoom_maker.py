# zoom and  E N H A N C E

import os
import shutil
import cv2
from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

# put your prompt for sampling here
my_prompt = "art by studio ghibli"
init_img = "endzoom.png"  # starting (or rather, ending image)
num_frames = 100000       # number of discrete zoom images to sample

mode = "zoom"
expand_softness = 50.
expand_space = 1.
expand_top = 50           # amount to expand in each direction in each step
expand_bottom = 50        # these values are in % of the original image size
expand_left = 50          # exceeding 50% in any direction is not recommended for recursive zooms / pans
expand_right = 50
""" # this will be added eventually, but I need to refactor some code first
mode = "pan"
expand_top = 0
expand_bottom = 0
expand_left = 0
expand_right = 50
"""

args = gdl.get_default_args() # sampling params
args.prompt = my_prompt
args.init_img = init_img
args.steps = 100 #100
args.scale = 14. #8.
args.guidance_strength = 0.35 #0.4 #0.25  # try lowering clip guidance_strength if you have problems with zooms "exploding"
args.noise_start = 2.     #1.618
#args.sampler = "k_euler_ancestral"

# *****************************************************************

expanded_img = os.path.splitext(init_img)[0]+".expanded.png"
cv2_image = cv2.imread(DEFAULT_PATHS.inputs+"/"+init_img)
original_size = cv2_image.shape

""" 
if mode == "pan":
    assert(((expand_left > 0) != (expand_right > 0)) or (expand_left == 0 and expand_right == 0))
    assert(((expand_top > 0) != (expand_bottom > 0)) or (expand_top == 0 and expand_bottom == 0))
"""

# create zoom frames
for i in range(num_frames):
    print("Starting iteration {0} of {1}...".format(i+1, num_frames))
    expand(args.init_img, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space, output_file=expanded_img, add_noise=2.)
    args.init_img = expanded_img
    args = sample(**vars(args))

    output_file = DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = DEFAULT_PATHS.inputs+"/"+args.init_img
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

    """
    if mode == "pan":
        cv2_image = cv2.imread(input_file)
        panX = max(int((expand_right - expand_left)/100. * cv2_image.shape[1] / 2.), 0)
        panY = max(int((expand_bottom - expand_top)/100. * cv2_image.shape[0] / 2.), 0)
        print(panX, panY)
        cv2_cropped = cv2_image[panY:original_size[0]//2+panY, panX:original_size[1]//2+panX]
        cv2.imwrite(input_file, cv2_cropped)
        print("Cropped {0} ...".format(input_file))
    """

print("Done!")