import random
from datasets import load_dataset



import torch

test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")

for p in range(10):
    print(test_dataset[p]["prompt"])