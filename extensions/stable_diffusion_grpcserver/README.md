An implementation of a server for the Stability AI API

# Installation

Option 1 (recommended):

Install Miniconda, then in a Conda console:

```
git clone https://github.com/hafriedlander/stable-diffusion-grpcserver.git
cd stable-diffusion-grpcserver
conda env create -f environment.yaml
conda activate sd-grpc-server
sets HF_API_TOKEN={your huggingface token}
python ./server.py
```

Option 2:

Create a directory and download https://raw.githubusercontent.com/hafriedlander/stable-diffusion-grpcserver/main/engines.yaml into it, then

```
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116 pip install stable-diffusion-grpcserver
set HF_API_TOKEN={your huggingface token} 
sdgrpcserver
```

# Features

- Txt2Img and Img2Img from Stability-AI/Stability-SDK, specifying a prompt
- Can load multiple pipelines, such as Stable and Waifu Diffusion, and swap between them as needed
- Mid and Low VRAM modes for larger generated images at the expense of some performance
- Adjustable NSFW behaviour
- Significantly enhanced masked painting:
  - When Strength < 1, uses normal diffusers inpainting (with improved mask gradient handling)
  - When Strength >= 1 and <= 2, uses seamless outpainting algorithm. 
    Strength above 1 acts as a boost - the higher the value, the more even areas protected by a mask are allowed to change
- Euler, Euler_A samplers are currently integrated, and DDIM accepts an ETA parameter
- Cancel over API (using GRPC cancel will abort the currently in progress generation)
- Negative prompting (send a `Prompt` object with `text` and a negative `weight`)


# Thanks to / Credits:

- Seamless outpainting https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2

# Roadmap

Core API functions not working yet:

- Some samplers (like dpm2) are not currently supported in Diffusers
- ChainGenerate not implemented

Extra features to add

- Progress reporting over the API is included but not exposed yet
- Embedding params in png
- Extra APIs
  - Image resizing
  - Aspect ratio shifting
  - Asset management
  - Extension negotiation so we can:
    - Ping back progress notices
    - Allow cancellation requests
    - Specify negative prompts
- CLIP guided generation https://github.com/huggingface/diffusers/pull/561
- Community features: 
  - Prompt calculation https://github.com/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb
  - Prompt suggestion https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion
  - Prompt compositing https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
  - Automasking https://github.com/ThstereforeGames/txt2mask
  - Other schedulers (needs custom pipeline for some). https://github.com/huggingface/diffusers/commit/489894e4d9272dec88fa3a64a8161aeab471fc18
  - Huge seeds
- Other thoughts
  - Figure out how to just suppress NSFW filtering altogether (takes VRAM, if you're not interested)