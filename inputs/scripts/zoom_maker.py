# zoom and  E N H A N C E

import glob
import shutil
import os
import numpy as np
from argparse import Namespace

args = cli_default_args()
args.prompt_style = "dreamlikeart, surreal, surrealism, Textless, Perfect ling, mucheneuve, muchenies ing, Jeffrigheme, realistacking, ejsin the bartgerm,  Perfect focus-stic listan, rhads, villess, colourealighelant lit"
args.zoom_prompt_schedule = [
    #"dreamlikeart, surreal, surrealism, Textless, Perfect ling, mucheneuve, muchenies ing, Jeffrigheme, realistacking, ejsin the bartgerm, Perfect focus-stic listan, rhads, villess, colourealighelant lit",
    #"dreamlikeart, abstract, surreal",
    "desert landscape with a watch buried in the sand",
    "a clocktower in the night sky",
    "an exotic bird with a crown on its head",
    "a huge apple tree with a snake wrapped around it",
    "a beautiful underwater city with colorful fish",
    "an open window with a beautiful view of the ocean",
    "a bed of hungry flowers",
    "golden stairs leading to a castle in the clouds",
    "a lavish banquet hall with a feast on the table",
    "a pile of large boulders",
    "a powerful river surrounded by trees", 
]
args.zoom_prompt_reset_interval = 1   # the prompt is switched to the next prompt in the list every n samples
args.zoom_prompt_schedule_order = "linear"  # rotate through the prompt list in order
#args.zoom_prompt_schedule_order = "random" # uncomment this line to use prompts in random order
args.zoom_interactive_cherrypicker = False  # setting this to True will prompt you to interactively accept or reject each keyframe / sample
                                            # currently broken until I can find a better way to distribute opencv2 with appropriate dependencies
args.num_samples = 1
args.zoom_num_frames = 1000 # number of discrete zoom images to sample
                            # (you can abort / close the program at any time to use the keyframes you have already generated)

args.expand_softness = 50.
args.expand_space = 10.     # distance to hard erase from source image edge
args.expand_top = 50        # amount to expand in each direction in each step
args.expand_bottom = 50     # these values are in % of the original image size
args.expand_left = 50       # exceeding 50% in any direction is not recommended for recursive zooms / pans
args.expand_right = 50

args.init_image = ""  # starting (or rather, ending image file, relative to inputs path). if blank start with a generated image
args.output_path = "zoom_maker"  # output path, relative to outputs
args.output_name = "zoom_maker"

#args.model_name = "stable-diffusion-v1-5-standard"
#args.steps = 80
#args.cfg_scale = 11. 
#args.guidance_strength = 0.

# if using sd2.x be sure to use a negative prompt
#args.model_name = "stable-diffusion-v2-1-standard"
#args.steps = 80
#args.cfg_scale = 6.#4.2
#args.guidance_strength = 0.

args.model_name = "dreamlike-diffusion-1.0"
args.steps = 80
args.cfg_scale = 11.
args.guidance_strength = 0.

args.negative_prompt = "frame, comic book, collage, cropped, oversaturated, signed, greyscale, monotone, vignette, title, text, logo, watermark"
#args.negative_prompt = "watermark, title, label, collage, cropped, highly saturated colors, monotone, vignette"
#args.negative_prompt = "art by lisa frank, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, text, watermark, disfigured, deformed, title, label, collage, vignette"
#args.negative_prompt = "frame, blender, cropped, lowres, poorly drawn face, poorly drawn hands, blurry, bad art, text, watermark, disfigured, deformed, title, label, collage, vignette"
#args.negative_prompt = "blender, lowres, poorly drawn face, blurry, bad art, text, watermark, disfigured, deformed, title, label, collage, vignette"

# these dims are only for the starting image (if no user-supplied init_image is used)
args.width = 768
args.height = 512
# for each subsequent generation the image is first expanded, but then contracted to fit inside the max width/height
args.max_width = 768
args.max_height = 512

args.hires_fix = False
args.sampler = "dpmspp_sde"
#args.sampler="dpmspp_2m"
#args.sampler = "k_euler_ancestral"
#args.sampler = "k_dpm_2_ancestral"
#args.sampler = "dpmspp_2"
#args.sampler = "dpmspp_3"
#args.sampler = "dpmspp_2s_ancestral"

# *****************************************************************

# if args or keyword args were passed in the cli run command, override the defaults
if cli_args: args = Namespace(**(vars(args) | vars(cli_args)))
if kwargs: args = Namespace(**(vars(args) | kwargs))

frame_filenames = sorted(glob.glob(gdl.DEFAULT_PATHS.outputs+"/"+args.output_path+"/*.png"), reverse=True)
if len(frame_filenames) > 0:
    args.init_image = "zoom_maker.png"
    output_file = frame_filenames[0]
    input_file = gdl.DEFAULT_PATHS.inputs+"/"+args.init_image
    print("Resuming zoom from {0}...".format(output_file))
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)
    start_frame_index = len(frame_filenames)
else:
    print("No frames found in {0} to resume from".format(gdl.DEFAULT_PATHS.outputs+"/"+args.output_path))
    if not args.init_image: print("No init_image specified, generating a random starting image...")
    else: print("Starting from '{0}'...".format(args.init_image))
    start_frame_index = 0

# create zoom frames
i = 0
while i < args.zoom_num_frames:
    print("Starting iteration {0} of {1}...".format(i+1, args.zoom_num_frames))

    # update the prompt according to the multi-prompt schedule
    if ((i % args.zoom_prompt_reset_interval) == 0) or (i == 0):
        if args.zoom_prompt_schedule_order == "linear":
            prompt_index = (i // args.zoom_prompt_reset_interval) % len(args.zoom_prompt_schedule)
        elif args.zoom_prompt_schedule_order == "random":
            prompt_index = np.random.randint(0, len(args.zoom_prompt_schedule))
        else:
            raise Exception("Unknown prompt schedule order '{0}'".format(args.zoom_prompt_schedule_order))
        args.prompt = args.zoom_prompt_schedule[prompt_index]
        if "prompt_style" in args: args.prompt = args.prompt_style + ", " + args.prompt
        print("prompt: {0}".format(args.prompt))

    args.img2img_strength = 2.
    args.output_name = "zoom_maker_f{0}".format(str(i+start_frame_index).zfill(4))
    sample(args)
    if args.status != 2: break # cancelled or error

    args.init_image = args.output_sample

    # currently disabled / broken because of opencv2 dependencies
    """
    if args.zoom_interactive_cherrypicker:
        input_key = 32
        while chr(input_key).lower() not in ("y","n"):
            cv2.imshow("Accept or reject? (y/n):", args.output_sample)
            input_key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if input_key == -1: break # window closed
        if input_key == -1: break     # terminate if window closed
        if chr(input_key).lower() != "y":
            output_file = gdl.DEFAULT_PATHS.outputs+"/"+args.output_file
            print("Removing {0} and retrying...".format(output_file))
            os.remove(output_file)
            continue
    """

    i += 1

print("Done!")