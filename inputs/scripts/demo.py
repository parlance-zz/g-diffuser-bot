sample("butter", seed=1, n=4, scale=10, output_path="compare1")
sample("butter", seed=1, n=4, scale=30, output_path="compare2")
compare("compare1", "compare2")
