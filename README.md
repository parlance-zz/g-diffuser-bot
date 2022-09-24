Copyright (c) 2022 Christopher Friesen

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


Sept 22/2022 Update - This is a temporary repository for breaking changes until they can be merged.
                    - This branch has the latest out-painting implementation.
                    - Major overhaul to how arguments, parameters, and input/output files are handled


Installation:
 - clone / download this repository to a folder of your choice
 - download / install miniconda (https://docs.conda.io/en/latest/miniconda.html)
 - open a conda prompt, then navigate to the folder where you cloned / downloaded this repository
 - conda env create -f environment.yaml
 - optional: edit g_diffuser_config.py and g_diffuser_defaults.py as appropriate, make sure to save!
 - place any pre-downloaded models into the models folder, if you want to use a hugging-face token instead enter it in g_diffuser_config.py
 
 Running:
 - open a conda prompt, then navigate to the g-diffuser folder
 - conda activate g_diffuser
 - optional: on Windows you can open prompt.bat to do the above automatically
 - run the discord bot with: python g_diffuser_bot.py
 - alternatively, run the CLI interface with: python g_diffuser_cli.py
 - You can use the CLI interface interactively with: python g_diffuser_cli.py --interactive
 - If you see an out of memory error use --use_optimized or change the default setting in g_diffuser_config.py
 - Verify your configuration by running: python g_diffuser_config.py or: pythong g_diffuser_defaults.py

Updating:
 - Simply download and replace your files with those from this repository. You probably won't need to replace your config and default settings files, but you may need to merge changes.
 
 
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
 Donations would also be greatly appreciated and will be used to fund further development.
 * https://buy.stripe.com/fZe8xU2lo0wU3SgfYY
 * ETH to 0x8e4BbD53bfF9C0765eE1859C590A5334722F2086
 
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 
 The algorithm is constantly evolving but some out-dated samples can be seen here:
 
    https://imgur.com/a/pwN6LHB
    
    https://imgur.com/a/S6g5SmI
    
    https://discord.gg/jS4vzBJxYz
    
    
 This section will be updated with better samples when I have time to create them.
