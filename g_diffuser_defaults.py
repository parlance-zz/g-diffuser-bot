import argparse

# ******************** SETTINGS BEGIN ************************

DEFAULT_SAMPLE_SETTINGS = argparse.Namespace()
DEFAULT_SAMPLE_SETTINGS.sampler = "k_euler"
DEFAULT_SAMPLE_SETTINGS.n = 1                        # number of samples to generate per command by default
DEFAULT_SAMPLE_SETTINGS.auto_seed_range = (1,9999)   # automatic seed range
DEFAULT_SAMPLE_SETTINGS.resolution = (512,512)       # default resolution for img / video outputs
DEFAULT_SAMPLE_SETTINGS.max_resolution = (1024,1024) # if you run out of ram due to resolution, try lowering this
DEFAULT_SAMPLE_SETTINGS.resolution_granularity = 8   # required by diffusers stable-diffusion for now
DEFAULT_SAMPLE_SETTINGS.strength = 0.42
DEFAULT_SAMPLE_SETTINGS.scale = 11.
DEFAULT_SAMPLE_SETTINGS.steps = 32
DEFAULT_SAMPLE_SETTINGS.noise_q = 1.
DEFAULT_SAMPLE_SETTINGS.model_name = "stable-diffusion-v1-4"

# ******************** SETTINGS END ************************

if __name__ == "__main__": # you can execute this file with python to see a summary of your defaults
    from g_diffuser_lib import print_namespace, get_args_parser
    print("\ndefault sample settings: ")
    print_namespace(DEFAULT_SAMPLE_SETTINGS, debug=True)
    
    parser = get_args_parser()
    args = parser.parse_args()
    print("\ndefault standard args: ")
    print_namespace(args, debug=True)
    
    from g_diffuser_bot import get_bot_args_parser
    bot_parser = get_bot_args_parser()
    bot_args = bot_parser.parse_args()
    print("\ndefault bot args: ")
    print_namespace(bot_args, debug=True)