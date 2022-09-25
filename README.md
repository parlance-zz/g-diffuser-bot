######  g-diffuser-lib ######

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Sept 24/2022 Update - What use to be the beta2 branch is finally merged to main as there were simply too many improvements and features
                      to avoid putting it off any longer. The discord bot is still under construction and will be operational again
                      very soon, but in the meantime please enjoy the new interactive CLI (it's fun to use, I promise!)
                      If you still need a functional SD discord bot with g-diffuser out-painting please check out the diffusers-beta branch.

- **an anaconda package for g-diffuser-lib will be published very soon**

Vision for the g-diffuser-lib project:
 - In the near future the diffusers library (https://github.com/huggingface/diffusers) being developed and promoted by hugging-face will expose multi-modality sampling abilities, meaning we will be able to arbitrarily mix and match input and output types. Tasks like txt2music, music2img, and everything in-between will be possible, and all of this will be coming within the next few months, not the next few years.
 - The goal of the project is to provide the best possible front-end, interface, and utilities for the diffusers library and to enable regular users to access these powerful abilities with a free and easy-to-use package that supports their local GPU and as many OS's / platforms as possible.
 - The current frontends include an (interactive) cli, http / json based command server, and a discord bot.
 - The current experimental extensions include g-diffuser fourier shaped noise out-painting.
 - For more information on progress and upcoming features please see: https://github.com/parlance-zz/g-diffuser-lib/issues
 
Development and funding:
 - Donations would also be greatly appreciated and will be directly used to fund further development.
 - https://buy.stripe.com/fZe8xU2lo0wU3SgfYY
 - ETH to 0x8e4BbD53bfF9C0765eE1859C590A5334722F2086

Installation:
 1)  clone this repository to a folder of your choice (or click the green "code" button up top and click "download zip")
 2)  download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 3)  open a conda prompt (click on the start menu and look for "anaconda prompt"),
     then navigate to the folder where you cloned / downloaded this repository.
 4)  run "conda env create -f environment.yaml"
 5)  place any pre-downloaded models into the models folder, if you want to use a hugging-face token instead, enter it in g_diffuser_config.py
     for specific instructions on model download / installation please see models/README.md (https://github.com/parlance-zz/g-diffuser-lib/tree/main/models)
 6)  If you are running Windows 10 you may need to turn on "developer mode". Look for "developer settings" in the start menu.
     
Optional: edit g_diffuser_config.py and g_diffuser_defaults.py and change settings as appropriate, save your changes
 
 Running:
 1)  open a conda prompt (click on the start menu and look for "anaconda prompt"), then navigate to the g-diffuser folder
 2)  run "conda activate g_diffuser" (OPTIONAL: on Windows you can open start_prompt.bat to do these 2 steps automatically)
 3)  run the discord bot with: "python g_diffuser_bot.py"
       - alternatively, run the CLI interface with: "python g_diffuser_cli.py"
       - You can use the CLI interface interactively with: "python g_diffuser_cli.py --interactive" (OPTIONAL: on Windows you can open start_interactive_cli.bat to open the interactive cli directly in one step)
       - If you see an out of memory error run: "python g_diffuser_cli.py --interactive --use-optimized"
       - Verify your configuration by running: "python g_diffuser_config.py" or: "python g_diffuser_defaults.py"

Updating:
 - Simply git pull or download and replace your files with those from this repository. You probably won't need to replace your g_diffuser_config.py or g_diffuser_defaults.py files, but you may need to merge changes.

Troubleshooting:
 - If you have questions or problems running anything in g-diffuser-lib, please post as much detailed information as you can in (https://github.com/parlance-zz/g-diffuser-lib/discussions/categories/q-a), myself or someone in the community may be able to help you. Thank you for your patience.
 
 
 G-Diffuser Experimental Fourier Shaped Noise In/out-painting Explanation:
 
  Why does this need to exist? I thought SD already did in/out-painting?:
 
 This seems to be a common misconception. Non-latent diffusion models such as Dall-e can be readily used for in/out-painting
 but the current SD in-painting pipeline is just regular img2img with a mask, and changing that would require training a
 completely new model (at least to my understanding). In order to get good results, SD needs to have information in the
 (completely) erased area of the image. Adding to the confusion is that the PNG file format is capable of saving color data in
 (completely) erased areas of the image but most applications won't do this by default, and copying the image data to the "clipboard"
 will erase the color data in the erased regions (at least in Windows). Code like this or patchmatch that can generate a
 seed image (or "fixed code") will (at least for now) be required for seamless out-painting.
 
 Although there are simple effective solutions for in-painting, out-painting can be especially challenging because there is no color data
 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.

 By taking a fourier transform of the unmasked source image we get a function that tells us the presence, orientation, and scale of features
 in that source. Shaping the init/seed/fixed code noise to the same distribution of feature scales, orientations, and positions/phases
 increases (visual) output coherence by helping keep features aligned and of similar orientation and size. This technique is applicable to any continuous
 generation task such as audio or video, each of which can be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased.
 TLDR: The fourier transform of the unmasked source image is a strong prior for shaping the noise distribution of in/out-painted areas
 
 For multi-channel data such as color or stereo sound the "color tone" of the noise can be bled into the noise with gaussian convolution and
 a final histogram match to the unmasked source image ensures the palette of the source is mostly preserved. SD is extremely sensitive to
 careful color and "texture" matching to ensure features are appropriately "bound" if they neighbor each other in the transition zone.
 
 The effects of both of these techiques in combination include helping the generator integrate the pre-existing view distance and camera angle,
 as well as being more likely to complete partially erased features (like appropriately completing a partially erased arm, house, or tree).
 
 Please note this implementation is written for clarity and correctness rather than performance.
 
 Todo: To be investigated is the idea of using the same technique directly in latent space. Spatial properties are (at least roughly?) preserved
 in latent space so the fourier transform should be usable there for the same reason convolutions are usable there. The ideas presented here
 could also be combined or augmented with other existing techniques.
 Todo: It would be trivial to add brightness, contrast, and overall palette control using simple parameters
 Todo: There are some simple optimizations that can increase speed significantly, e.g. re-using FFTs and gaussian kernels

 This code is provided under the MIT license -  Copyright (c) 2022 Christopher Friesen
 To anyone who reads this I am seeking employment in related areas.
 
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 
 The algorithm is constantly evolving but some out-dated samples can be seen here:
 
    https://imgur.com/a/pwN6LHB
    
    https://imgur.com/a/S6g5SmI
    
    https://discord.gg/jS4vzBJxYz
    
    
 This section will be updated with better samples when I have time to create them.
