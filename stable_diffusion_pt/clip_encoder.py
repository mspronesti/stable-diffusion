import torch
from torch import nn

from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_value = nn.Parameter(torch.zeros((n_token, n_embed)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_value
        return x


class CLIPEncoderLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # quick gelu
        x *= torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += residue

        return x


class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPEncoderLayer(12, 768) for _ in range(12)])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)
        # apply all the CLIP layers sequentially
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
