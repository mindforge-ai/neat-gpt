from torch import nn
from .decoder import Decoder


class GPT(nn.Module):
    def __init__(
        self,
        num_layers=12,
        embedding_dim=1024,
        num_attention_heads=16,
        context_len=512,
        vocab_len=50257,
        attention_dropout=0.5,
        outwards_dropout=0.5,
    ):
        super().__init__()
        self.decoder = Decoder(num_layers, embedding_dim, num_attention_heads, context_len, vocab_len, attention_dropout, outwards_dropout)
        self.clm_head = nn.Linear(embedding_dim, vocab_len, bias=False)
        """ self.clm_head.weight = (
            self.decoder.token_embedding.weight
        )  # weight tying, although I don't think this works fully. """

        num_parameters = sum(p.numel() for p in self.parameters())
        print(f"{num_parameters} parameters")

    def forward(self, X):
        hidden_states = self.decoder(X)
        logits = self.clm_head(hidden_states)
        return logits
