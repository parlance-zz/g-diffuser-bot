import argparse

# --- bot params and defaults -----

DEFAULT_SAMPLE_SETTINGS = argparse.Namespace()
DEFAULT_SAMPLE_SETTINGS.auto_seed_range = (1,9999)
DEFAULT_SAMPLE_SETTINGS.resolution = (512,512)
DEFAULT_SAMPLE_SETTINGS.max_resolution = (768,768)
DEFAULT_SAMPLE_SETTINGS.resolution_granularity = 64 # required by diffusers stable-diffusion for now
DEFAULT_SAMPLE_SETTINGS.strength = 0.42
DEFAULT_SAMPLE_SETTINGS.scale = 11.
DEFAULT_SAMPLE_SETTINGS.steps = 32