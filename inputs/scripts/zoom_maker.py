# zoom and  E N H A N C E

import glob
import shutil
import numpy as np
from argparse import Namespace

recycle_image = "zoom_maker.png" # file path in /inputs/ to copy over from outputs for recursive zoom

args = cli_default_args()
args.zoom_prompt_schedule = [
    #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed fantasy, trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
    #"a wooden shed, many green plant and flower gowing on it, illustration key visual trending pixiv fanbox by wlop and greg rutkowski and makoto shinkai and studio ghibli",
    #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed fantasy, rich colors, high contrast, gloomy atmosphere, dark background. trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
    #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed science fiction, rich colors, high contrast, gloomy atmosphere, dark background. trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
    #"entrance of the huge castle surrounded by advanced brutalist architecture, art by moebius, irakli nadar, overdetailed art, colorful, artistic record jacket design",
    #"Shattered glass retro stylized, cracked obsidian geometric fantasy art, splattering black gold iridescent bubbly underwater scenery, greg rutkowski, lois van baarle, ilya kuvshinov",
    #"Deep space, psychedelic, organic, oni compound artwork, artstation, beeple, mf marling fantasy epcot, glitchcore portrait of omin dran mind flayer psion politician, key portrait realism, druid octane trending gems, octane organic cinematic, mindar mumford, helmet, high character, futurescape, style final unreal of punk, mindar punk, borne space library artwork",
    #"Deep space, octane organic cinematic, mindar mumford, helmet, high character, futurescape, style final unreal of punk, mindar punk, borne space library artwork",
    #"Deep space, mindar mumford, helmet, high character, futurescape, style final unreal of punk, mindar punk, borne space library artwork",
    #"glitchcore portrait of omin dran mind flayer psion politician, key portrait realism, druid octane trending gems, octane organic cinematic, mindar mumford, helmet, high character, futurescape, style final unreal of punk, mindar punk, borne space library artwork",

    #"cyberpunk, city, street art, graffiti, spraypaint, colored smoke, heavily stylized, surreal, surrealism, artstation, beeple, key portrait realism, organic cinematic, mindar mumford, helmet, high character, futurescape, style final unreal of punk, mindar punk, borne space library artwork",
    #"epiphanies, stone figures and statues, natural warm spring lit by strong sunlight shining through, vast, wide angle, waterfall, close up portrait, detailed, photo, photorealistic, hyperrealistic",
    #"mopaumowhe intufinism",
    "Textless, Perfect ling, mucheneuve, muchenies ing, Jeffrigheme,  realistacking, ejsin the bartgerm,  Perfect focus-stic listan, rhads, villess, colourealighelant lit",
]
args.zoom_prompt_reset_interval = 2   # the prompt is switched to the next prompt in the list every n samples
#args.zoom_prompt_schedule_order = "linear"  # rotate through the prompt list in order
args.zoom_prompt_schedule_order = "random" # uncomment this line to use prompts in random order

args.num_samples = 1
args.zoom_num_frames = 1000 # number of discrete zoom images to sample
                            # (you can abort / close the program at any time to use the keyframes you have already generated)

args.expand_softness = 15.
args.expand_space = 1.      # distance to hard erase from source image edge
args.expand_top = 25        # amount to expand in each direction in each step
args.expand_bottom = 25     # these values are in % of the original image size
args.expand_left = 25       # exceeding 50% in any direction is not recommended for recursive zooms / pans
args.expand_right = 25

args.init_image = ""  # starting (or rather, ending image file, relative to inputs path). if blank start with a generated image
args.output_path = "zoom_maker"  # output path, relative to outputs
args.output_name = "zoom_maker"
args.steps = 11 #8 #60
args.cfg_scale = 10.5 #12.
args.guidance_strength = 1. #0.7 #0.5   # try lowering clip guidance_strength if you have problems with zooms "exploding"
#args.negative_prompt = "frame, comic book, collage, cropped, oversaturated, signed, greyscale, monotone, vignette, title, text, logo, watermark"
args.negative_prompt = "watermark, title, label, collage, cropped, highly saturated colors, monotone, vignette"
#args.sampler = "k_euler_ancestral"
#args.sampler = "k_dpm_2_ancestral"
#args.sampler="plms"
#args.sampler = "dpmspp_2"
args.sampler = "dpmspp_3"


# *****************************************************************

# if args or keyword args were passed in the cli run command, override the defaults
if cli_args: args = Namespace(**(vars(args) | vars(cli_args)))
if kwargs: args = Namespace(**(vars(args) | kwargs))

frame_filenames = sorted(glob.glob(gdl.DEFAULT_PATHS.outputs+"/"+args.output_path+"/*.png"), reverse=True)
if len(frame_filenames) > 0:
    args.init_image = recycle_image
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
for i in range(args.zoom_num_frames):
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
        print("prompt: {0}".format(args.prompt))

    args.output_name = "zoom_maker_f{0}".format(str(i+start_frame_index).zfill(4))
    args = sample(args)
    if args.status != 2: break # cancelled or error

    # recycle the output back into the next input init_image for a recursive zoom effect
    args.init_image = recycle_image
    output_file = gdl.DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = gdl.DEFAULT_PATHS.inputs+"/"+args.init_image
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

print("Done!")