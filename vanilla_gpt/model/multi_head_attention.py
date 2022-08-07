import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        assert (
            self.embedding_dim % self.num_attention_heads == 0
        ), "embedding_dim must be divisible by num_attention_heads"
        self.embedding_dim_per_attention_head = (
            self.embedding_dim // self.num_attention_heads
        )

        self.to_queries_keys_values = nn.Linear(
            config["embedding_dim"], 3 * config["embedding_dim"]
        )
        self.outwards = nn.Linear(config["embedding_dim"], config["embedding_dim"])

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config["context_len"],
                    config["context_len"],
                ).view(1, 1, config["context_len"], config["context_len"])
            ),
        )

    def forward(self, X):
        (
            batch_len,
            seq_len,
            _,
        ) = (
            X.size()
        )  # at inference time the sequence length might not be the full context length

        queries_keys_values = self.to_queries_keys_values(X)
        queries, keys, values = queries_keys_values.split(self.embedding_dim, dim=2)

        queries = queries.view(
            batch_len,
            seq_len,
            self.num_attention_heads,
            self.embedding_dim_per_attention_head,
        ).transpose(1, 2)
        keys = keys.view(
            batch_len,
            seq_len,
            self.num_attention_heads,
            self.embedding_dim_per_attention_head,
        ).transpose(1, 2)
        values = values.view(
            batch_len,
            seq_len,
            self.num_attention_heads,
            self.embedding_dim_per_attention_head,
        ).transpose(1, 2)

        scaled_scores = torch.matmul(queries, keys.transpose(-2, -1)) * (
            1 / math.sqrt(self.embedding_dim)
        )
        masked_scores = scaled_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        softmaxed_scores = F.softmax(masked_scores, dim=-1)
        attention_values = torch.matmul(softmaxed_scores, values)
        attention_values = (
            attention_values.transpose(1, 2)
            .contiguous()
            .view(batch_len, seq_len, self.embedding_dim)
        )
        outputs = self.outwards(attention_values)
        return outputs
