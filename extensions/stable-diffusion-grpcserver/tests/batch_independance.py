
import os, sys, re, time, random
from types import SimpleNamespace as SN

import torch

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

basePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(basePath)

from sdgrpcserver.server import main
from sdgrpcserver.services.generate import GenerationServiceServicer
from sdgrpcserver.manager import EngineMode, EngineManager
from sdgrpcserver import images

import generation_pb2, generation_pb2_grpc

def str_to_seed(s):
    n = random.Random(s).randint(0, 2**32 - 1)
    while n >= 2**32:
        n = n >> 32
    return n 

with open(os.path.normpath("testengines.yaml"), 'r') as cfg:
    engines = yaml.load(cfg, Loader=Loader)

manager = EngineManager(
    engines, 
    weight_root="../weights/",
    mode=EngineMode(vram_optimisation_level=2, enable_cuda=True, enable_mps=False), 
    nsfw_behaviour="flag"
)

manager.loadPipelines()
pipeline = manager.getPipe("stable-diffusion-v1-4")

params=SN(
    height=512,
    width=512,
    cfg_scale=7.5,
    eta=0,
    sampler=generation_pb2.SAMPLER_DDIM,
    steps=50,
    seed=[420420420, 420420421],
    samples=4,
    strength=0.8
)

def save(prefix, results):
    ctr=0
    for result_image, nsfw in zip(results[0], results[1]):
        binary=images.toPngBytes(result_image)[0]
        with open(f"out/{prefix}-{ctr}.png", "wb") as f:
            f.write(binary)
            ctr += 1


with open("image.png", "rb") as file:
    test_image = file.read()
    image = images.fromPngBytes(test_image).to(manager.mode.device)

with open("mask.png", "rb") as file:
    test_mask = file.read()
    mask = images.fromPngBytes(test_mask).to(manager.mode.device)

# -- Check img2img

params.seed = [420420420, 420420421]
save("batched", pipeline.generate(["A Crocodile", "A Crocodile"], params))

params.seed = 420420420
save("nobatch0", pipeline.generate("A Crocodile", params))

params.seed = 420420421
save("nobatch1", pipeline.generate("A Crocodile", params))

# -- Check img2img

params.seed = [420420420, 420420421]
save("batched-i2i", pipeline.generate(["A Crocodile", "A Crocodile"], params, image=image))

params.seed = 420420420
save("nobatch0-i2i", pipeline.generate("A Crocodile", params, image=image))

params.seed = 420420421
save("nobatch1-i2i", pipeline.generate("A Crocodile", params, image=image))

# -- Check inpaint

params.strength=1
params.seed = [420420420, 420420421]
save("batched-ip", pipeline.generate(["A Crocodile", "A Crocodile"], params, image=image, mask=mask))

params.seed = 420420420
save("nobatch0-ip", pipeline.generate("A Crocodile", params, image=image, mask=mask))

params.seed = 420420421
save("nobatch1-ip", pipeline.generate("A Crocodile", params, image=image, mask=mask))

