sample("butter", seed=1, num_samples=4, cfg_scale=10, output_path="compare1")
sample("butter", seed=1, num_samples=4, cfg_scale=30, output_path="compare2")
compare("compare1", "compare2")

# if things get a bit cluttered you can put arguments on their own line
sample(
    "more_butter",
    cfg_scale=30,
    seed=10,
    sampler="k_euler_ancestral",
)
