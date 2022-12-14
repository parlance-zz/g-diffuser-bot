# zoom and  E N H A N C E

import glob
import shutil
import os
import numpy as np
from argparse import Namespace

args = cli_default_args()
args.zoom_prompt_schedule = [
    #"Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed science fiction, rich colors, high contrast, gloomy atmosphere, dark background. trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
    #"Textless, Perfect ling, mucheneuve, muchenies ing, Jeffrigheme,  realistacking, ejsin the bartgerm,  Perfect focus-stic listan, rhads, villess, colourealighelant lit",
    #"a couple of statues sitting next to each other in a building, by Jeremy Geddes, Artstation contest winner, gothic art, wreathed in flame, shattered abstractions, book cover illustration, gothic wraith maiden, peter mohrbacher c 2 0, ffffound, loss in despair, photoshop art, ornate and flowing, promo image, whirling death",
    #"a very strange looking creature in the middle of a desert, cyberpunk art, by Mike 'Beeple' Winkelmann, fantasy art, many mech flowers, high detail 8k render, portrait of a zentaur, cute detailed digital art, robot dragon head, cyberpunk in foliage, behance. high detail, vaporwave artwork composition, beautiful avatar pictures",

    #"a painting of bench in front of a window, a surrealist painting, by Justin Gerard, behance contest winner, paul lehr and m. c. escher, very very well detailed image, underground room, charles burns, shusei nagaoka, magical portal opened, mysterious laboratory, establishing shot, chris moore",
    #"a man standing on top of a hill next to a lake, concept art, psychedelic art, colorful dark vector, rich deep colors. masterpiece, dark neon colored universe, michael page, style of kilian eng, high detailed painting, awesome greate composition, lowres, abstract surrealist, journey, fantasy sticker illustration",
    #"a painting of a man with a bird in his hand, poster art, by Jason Benjamin, cg society contest winner, psychedelic art, with a glass eye and a top hat, in a deep lush jungle at night, pinguin, dan mumford and thomas kinkade, bubbling ooze covered serious, promo shot, black tie, disney artist, album artwork, booze",
    #"a painting of a strange creature with a giant eye, by Justin Gerard, cg society contest winner, psychedelic art, lush alien landscape, album artwork, black ooze, detailed wide shot, thomas kinkade and tristan eaton, river styx, very very well detailed image, a dream, brock hofer, caravan, hollow, painted in high resolution, mind-bending",
    #"a painting of close-up portrait of a man, a detailed painting, by Justin Gerard, Artstation contest winner, psychedelic art, glowing eyes in helmet, medium portrait of a goblin, beeple and jeremiah ketner, majora mask, highly detailed zen neon, official fanart behance hd, sofubi, detailed mask, dmt temple, trailer",
    #"art by Moooooebius, close up portrait, color, surreal, surrealism",
    #"a close up of a person with a beard and glasses, cyberpunk art, by Beeple, digital art, sitting on santa, unsplash photo contest winner, karthus from league of legends, kerem beyit, tabletop gaming, hyper real photo, duke nukem, hide the pain harold, dazzling lights, baron harkonnen, toy photography, profile pic",
    "christmas, isometric view of festive Skyrim whiterun art by thomas kinkade, A Winter Scene with a child near a Castle by Hendrick Avercamp, snowing, christmas trees, festive winter decorations, christmas",
]
args.zoom_prompt_reset_interval = 1   # the prompt is switched to the next prompt in the list every n samples
#args.zoom_prompt_schedule_order = "linear"  # rotate through the prompt list in order
args.zoom_prompt_schedule_order = "random" # uncomment this line to use prompts in random order
args.zoom_interactive_cherrypicker = False  # setting this to True will prompt you to interactively accept or reject each keyframe / sample
                                            # currently broken until I can find a better way to distribute opencv2 with appropriate dependencies
args.num_samples = 1
args.zoom_num_frames = 1000 # number of discrete zoom images to sample
                            # (you can abort / close the program at any time to use the keyframes you have already generated)

args.expand_softness = 25.
args.expand_space = 10.     # distance to hard erase from source image edge
args.expand_top = 30        # amount to expand in each direction in each step
args.expand_bottom = 30     # these values are in % of the original image size
args.expand_left = 30       # exceeding 50% in any direction is not recommended for recursive zooms / pans
args.expand_right = 30

args.init_image = ""  # starting (or rather, ending image file, relative to inputs path). if blank start with a generated image
args.output_path = "zoom_maker"  # output path, relative to outputs
args.output_name = "zoom_maker"

args.model_name = "stable-diffusion-v1-5-standard"
args.steps = 80
args.cfg_scale = 11. 
args.guidance_strength = 0.25

# if using sd2.x be sure to use a negative prompt
#args.model_name = "stable-diffusion-v2-1-standard"
#args.steps = 80
#args.cfg_scale = 4.2
#args.guidance_strength = 0.

#args.negative_prompt = "frame, comic book, collage, cropped, oversaturated, signed, greyscale, monotone, vignette, title, text, logo, watermark"
#args.negative_prompt = "watermark, title, label, collage, cropped, highly saturated colors, monotone, vignette"
#args.negative_prompt = "art by lisa frank, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, text, watermark, disfigured, deformed, title, label, collage, vignette"
#args.negative_prompt = "frame, blender, cropped, lowres, poorly drawn face, poorly drawn hands, blurry, bad art, text, watermark, disfigured, deformed, title, label, collage, vignette"
#args.negative_prompt = "blender, lowres, poorly drawn face, blurry, bad art, text, watermark, disfigured, deformed, title, label, collage, vignette"

# these dims are only for the starting image (if no user-supplied init_image is used)
args.width = 896
args.height = 512
# for each subsequent generation the image is first expanded, but then contracted to fit inside the max width/height
args.max_width = 896
args.max_height = 512

args.hires_fix = True
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