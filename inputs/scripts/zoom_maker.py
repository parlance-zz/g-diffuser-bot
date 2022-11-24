# zoom and  E N H A N C E

import os
import shutil
import cv2
import numpy as np

from g_diffuser_defaults import DEFAULT_SAMPLE_SETTINGS
from g_diffuser_config import DEFAULT_PATHS

# put your prompt for sampling here
my_prompts = [
    "Face portrait of a retrofuturistic assassin surrounded by advanced brutalist architecture. highly detailed fantasy, trending on artstation an ultrafine hyperdetailed colorfull illustration by kim jung gi, moebius, irakli nadar, alphonse mucha, ayami kojima, amano, greg hildebrandt",
    "Isometric view of beautiful! epic sci fi building landscape by james gurney and simon stalenhag, wlop, artgerm, yasutomo oka, yuumei, rendered in unreal engine and redshift octane, background is an, digital art dynamic dramatic lighting, bokeh, imagine fx, artstation, cgsociety, zbrush central, by bandai namco artist, macross frontier, dynamic camera angle, art deco patterns",
    "a wooden shed, many green plant and flower gowing on it, illustration concept art anime key visual trending pixiv fanbox by wlop and greg rutkowski and makoto shinkai and studio ghibli",
    "face portrait clockwork steampunk helmet mask robot ninja stylized digital illustration sharp focus surrounded by advanced brutalist architecture, elegant intricate digital painting artstation concept art global illumination ray tracing advanced technology chaykin howard and campionpascale and cooke darwyn and davis jack",
    "celestial beings, futuristic by filipe pagliuso and justin gerard symmetric fantasy highly detailed realistic intricate port",
    "Hyperrealistic portrait of a woman monster astronaut, full body portrait, well lit, intricate abstract, cyberpunk, intricate artwork, by tooth wu, wlop, beeple, octane render, in the style of jin kagetsu, james jean and wlop, highly detailed, sharp focus, intricate concept art, ambient lighting, artstation",
    "big giant moon in the background, side portrait dark witch, witch outfit large cloak, fantasy forest landscape, dragon scales, fantasy magic, undercut hairstyle, short purple black fade hair, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, matte, art by wlop and artgerm and greg rutkowski and alphonse mucha, masterpiece",
    "a beautiful venus monster astronaut defined facial features, intricate abstract. cyberpunk, symmetrical facial features. biomechanics. beautiful intricately detailed japanese crow kitsune mask and clasical japanese kimono. betta fish, jellyfish phoenix, bio luminescent, plasma, by ruan jia and artgerm and range murata and wlop and ross tran and william, adolphe bouguereau and beeple, key art, award winning, artstation, intricate details, realistic, hyperdetailed",
    "a starship floating in space near a gas giant with an undersized stone devil's tower integrated into the structure of the ship, cinematic lighting, detailed, cell shaded, warm colours, by wlop, ilya kuvshinov, artgerm, krenz cushart, greg rutkowski, cinematic dramatic atmosphere, sharp focus, volumetric lighting, cinematic lighting, studio quality",
    "a huge humanoid robot in a park, sunset, a dark dystopian city behind a huge wall, dystopian, stunning, cinematic lighting, concept art by greg rutkowski and simon stalenhag and wlop, artstation",
    "Ground floor of secret overwatch common area carved inside a cave, doors to various living quarters, magical, natural light, huge central tree, flowers, clean lines, cozy, fantasy, minimalist, clean lines, architecture, sharp focus, concept art, artstation, by greg rutkowski",
    "A solitary figure in a misty japanese bamboo forest, cell shaded, huge waterfall, large rocky mountain, drawing, stylized anime, sun rays, soft, by hayao miyazaki, ghibli studio, makoto shinkai, toei animation, studio trigger, trending on artstation, hd",
]

init_img = "endzoom.png"  # starting (or rather, ending image)
num_frames = 100000       # number of discrete zoom images to sample

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
args.steps = 100  #100
args.scale = 12. #14. #8.
args.guidance_strength = 0.5 #0.4 #0.3 #0.25  # try lowering clip guidance_strength if you have problems with zooms "exploding"
args.noise_start = 2.
args.negative_prompt = "border, frame, comic book, collage, cropped, oversaturated, signed, greyscale, monotone, vignette, title, poster, text, logo, watermark"

#args.sampler = "k_euler_ancestral"

# *****************************************************************

expanded_img = os.path.splitext(init_img)[0]+".expanded.png"
cv2_image = cv2.imread(DEFAULT_PATHS.inputs+"/"+init_img)
original_size = cv2_image.shape

# create zoom frames
for i in range(num_frames):
    print("Starting iteration {0} of {1}...".format(i+1, num_frames))
    expand(args.init_img, expand_top, expand_right, expand_bottom, expand_left, expand_softness, expand_space, output_file=expanded_img, add_noise=1.)

    args.prompt = my_prompts[np.random.randint(0, len(my_prompts))]
    print("prompt: {0}".format(args.prompt))
    args.init_img = expanded_img
    args = sample(**vars(args))

    output_file = DEFAULT_PATHS.outputs+"/"+args.output_file
    input_file = DEFAULT_PATHS.inputs+"/"+args.init_img
    print("Copying from {0} to {1}...".format(output_file, input_file))
    shutil.copyfile(output_file, input_file)

print("Done!")