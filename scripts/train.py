from torch import embedding
from vanilla_gpt import GPT
import torch

config = {"num_layers": 12, "embedding_dim": 500, "seq_len": 200, "vocab_len": 50000}

model = GPT(config)
print(model)

test = torch.randint(2, 5000, [3, 200])
print(test)
print(model(test))
