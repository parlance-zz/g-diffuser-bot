import argparse

# ******************** SETTINGS BEGIN ************************

DEFAULT_SAMPLE_SETTINGS = argparse.Namespace()
DEFAULT_SAMPLE_SETTINGS.sampler = "k_euler"                  # default sampling mode (ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_lms)
DEFAULT_SAMPLE_SETTINGS.n = 1                                # number of samples to generate per sample command by default
DEFAULT_SAMPLE_SETTINGS.resolution = (512,512)               # default resolution for img / video outputs
DEFAULT_SAMPLE_SETTINGS.max_resolution = (1280,1280)         # if you run out of ram due to output resolution lower this to prevent exceeding your max memory
DEFAULT_SAMPLE_SETTINGS.resolution_granularity = 64          # required by diffusers stable-diffusion for now due to latent space subsampling
DEFAULT_SAMPLE_SETTINGS.noise_start = 0.65                   # default strength for pure img2img or style transfer
DEFAULT_SAMPLE_SETTINGS.noise_end = 0.01                     # can be used to influence in/out-painting quality
DEFAULT_SAMPLE_SETTINGS.noise_eta = 0.70                     # can be used to influence in/out-painting quality
DEFAULT_SAMPLE_SETTINGS.scale = 10.                          # default cfg scale
DEFAULT_SAMPLE_SETTINGS.steps = 50                           # default number of sampling steps, lower to reduce sampling time
DEFAULT_SAMPLE_SETTINGS.noise_q = 1.                         # fall-off of shaped noise distribution for in/out-painting
DEFAULT_SAMPLE_SETTINGS.auto_seed_range = (10000,99999)      # automatic random seed range
DEFAULT_SAMPLE_SETTINGS.model_name = "stable-diffusion-v1-5" # default model id to use, see g_diffuser_config_models.yaml for the list of models to be loaded by grpc server
DEFAULT_SAMPLE_SETTINGS.guidance_strength = 0.25             # default guidance strength (only affects 'clip guided' models)
DEFAULT_SAMPLE_SETTINGS.negative_prompt = None               # default negative prompt string or None

# ******************** SETTINGS END ************************

if __name__ == "__main__": # you can execute this file with python to see a summary of your defaults
    from extensions.g_diffuser_lib import print_namespace, get_args_parser
    print("\ndefault sample settings: ")
    print_namespace(DEFAULT_SAMPLE_SETTINGS, debug=True)
    
    parser = get_args_parser()
    args = parser.parse_args()
    print("\ndefault standard args: ")
    print_namespace(args, debug=True)