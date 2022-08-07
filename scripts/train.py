from vanilla_gpt import GPT
import torch
import torch.nn.functional as F
from rich.progress import track

config = {
    "num_layers": 12,
    "embedding_dim": 128,
    "num_attention_heads": 16,
    "context_len": 2048,
    "vocab_len": 50257,
}

model = GPT(config).to(0)

dataset = torch.load("./example.pt")

opt = torch.optim.SGD(lr=0.01, params=model.parameters())

model.train()

for sequence in track(dataset):
    opt.zero_grad()
    raw = sequence
    source = raw[: config["context_len"]].to(torch.int32)
    source_len = source.size(0)
    pad_len = config["context_len"] - source_len
    if pad_len > 0:
        source = torch.cat(
            [source, torch.zeros([1, pad_len], dtype=torch.int32)], dim=1
        )
    source = source.to(0).unsqueeze(0)
    output = model(source)
    target = raw[1 : config["context_len"] + pad_len + 1]
    target = target.to(0).unsqueeze(0)
    loss = F.cross_entropy(
        output.flatten(0, -2),
        target.flatten(),
        reduction="mean",
    )

    loss.backward()
    print(loss.item())
    opt.step()
