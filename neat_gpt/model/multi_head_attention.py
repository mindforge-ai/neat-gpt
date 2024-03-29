import math
import torch
from torch import nn
import torch.nn.functional as F
from .utils import Conv1D


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_attention_heads,
        context_len,
        attention_dropout,
        outwards_dropout,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        assert (
            self.embedding_dim % self.num_attention_heads == 0
        ), "embedding_dim must be divisible by num_attention_heads"
        self.embedding_dim_per_attention_head = (
            self.embedding_dim // self.num_attention_heads
        )

        """ self.to_queries_keys_values = Conv1D(
            3 * self.embedding_dim, 1, self.embedding_dim
        )
        self.outwards = Conv1D(self.embedding_dim, 1, self.embedding_dim) """

        self.to_queries_keys_values = nn.Linear(
            self.embedding_dim, 3 * self.embedding_dim
        )
        self.outwards = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_len,
                    context_len,
                ).view(1, 1, context_len, context_len)
            ),
        )

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.outwards_dropout = nn.Dropout(outwards_dropout)

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

        scaled_scores = torch.matmul(queries, keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))

        masked_scores = scaled_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        dropped_scores = masked_scores  # self.attention_dropout(masked_scores) dropout here not working
        softmaxed_scores = F.softmax(dropped_scores, dim=-1)
        attention_values = torch.matmul(softmaxed_scores, values)
        attention_values = (
            attention_values.transpose(1, 2)
            .contiguous()
            .view(batch_len, seq_len, self.embedding_dim)
        )  # merge heads

        outputs = self.outwards(attention_values)
        outputs = self.outwards_dropout(outputs)
        return outputs
