from neat_gpt import GPT
import torch
from rich.progress import track
import wandb
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F

model_config = {
    "num_layers": 12,
    "embedding_dim": 1024,
    "num_attention_heads": 16,
    "context_len": 511,
    "vocab_len": 50257,
    "attention_dropout": 0.5,
    "outwards_dropout": 0.5,
    "num_epochs": 100,
    "batch_size": 2
}

model = GPT(model_config).to(0)

class BooksCorpus(Dataset):
    def __init__(self, directory):
        self.data = torch.load("../data/books1/bookscorpus-2.pt")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

training_data = BooksCorpus(None)
train_dataloader = DataLoader(training_data, batch_size=model_config["batch_size"], shuffle=True)

opt = torch.optim.AdamW(model.parameters(), lr=0.00025)

# original code uses tf.nn.sparse_softmax_cross_entropy_with_logits;

model.train()

# wandb.init(project="neat-gpt", config=model_config, entity="mindforge-ai")

for epoch in range(model_config["num_epochs"]):
    for batch in track(train_dataloader):
        opt.zero_grad()
        raw = batch
        source = raw[:, : model_config["context_len"]]
        source = source.to(0)
        output = model(source) # (batch_len, seq_len, vocab_len)
        target = raw[:, 1:model_config["context_len"] + 1] # (batch_len, seq_len)
        target = target.to(0).to(torch.int64)

        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1)) # collapse batch_len and seq_len dimensions to batch_len * seq_len
        wandb.log({"loss": loss.item()})

        loss.backward()
        opt.step()
