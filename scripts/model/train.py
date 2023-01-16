from neat_gpt import GPT
import torch
from rich.progress import track
import wandb
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
import argparse

model_config = {
    "num_layers": 12,
    "embedding_dim": 1024,
    "num_attention_heads": 16,
    "context_len": 511,
    "vocab_len": 50257,
    "attention_dropout": 0.5,
    "outwards_dropout": 0.5,
    "num_epochs": 100,
    "batch_size": 12,
    "learning_rate": 0.00025,
}


class BooksCorpus(Dataset):
    def __init__(self, directory):
        self.data = torch.load("../data/books1/bookscorpus-2.pt")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=511)
    parser.add_argument("--vocab-len", type=int, default=50257)
    parser.add_argument("--attention-dropout", type=float, default=0.5)
    parser.add_argument("--outwards-dropout", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.000025)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    model = GPT(
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        context_len=args.context_len,
        vocab_len=args.vocab_len,
        attention_dropout=args.attention_dropout,
        outwards_dropout=args.outwards_dropout,
    ).to(0)

    training_data = BooksCorpus(None)
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # original code uses tf.nn.sparse_softmax_cross_entropy_with_logits;

    model.train()

    if args.log:
        wandb.init(project="neat-gpt", config=model_config, entity="mindforge-ai")

    for epoch in range(model_config["num_epochs"]):
        for batch in track(train_dataloader):
            opt.zero_grad()
            source = batch[:, : args.context_len]
            source = source.to(0).to(torch.int64)
            logits = model(source)  # (batch_len, seq_len, vocab_len)
            target = batch[:, 1 : args.context_len + 1]  # (batch_len, seq_len)
            target = target.to(0).to(torch.int64)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1
            )  # collapse batch_len and seq_len dimensions to batch_len * seq_len
            if args.log:
                wandb.log({"loss": loss.item()})

            loss.backward()
            opt.step()
