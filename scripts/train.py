from vanilla_gpt import GPT
import torch
import torch.nn.functional as F
from rich.progress import track

model_config = {
    "num_layers": 12,
    "embedding_dim": 1024,
    "num_attention_heads": 16,
    "context_len": 2048,
    "vocab_len": 50257,
}

training_config = {"use_wandb": False}

model = GPT(model_config).to(0)

dataset = torch.load("./example.pt")

opt = torch.optim.SGD(lr=0.01, params=model.parameters())

# original code uses tf.nn.sparse_softmax_cross_entropy_with_logits;
loss_function = torch.nn.CrossEntropyLoss()

model.train()

if training_config["use_wandb"]:
    import wandb

    wandb.init(project="gpt")

for sequence in track(dataset):
    opt.zero_grad()
    raw = sequence
    source = raw[: model_config["context_len"]].to(torch.int32)
    source_len = source.size(0)
    pad_len = model_config["context_len"] - source_len
    if pad_len > 0:
        source = torch.cat(
            [source, torch.zeros([1, pad_len], dtype=torch.int32)], dim=1
        )
    source = source.to(0).unsqueeze(0)
    output = model(source)
    target = raw[1 : model_config["context_len"] + pad_len + 1]
    target = target.to(0).unsqueeze(0)

    # tensor views are used to stack all batches and perform cross entropy across all logits and targets
    # but I wonder if there is a loss of useful data here: are non-argmaxed logit values compared with 0 in the target?
    loss = loss_function(output.view(-1, model_config["vocab_len"]), target.view(-1))

    if training_config["use_wandb"]:
        wandb.log({"loss": loss.item()})

    loss.backward()
    opt.step()
