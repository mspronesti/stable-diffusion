import torch
from torch import nn
from .attention import SelfAttention
from .base import PreTrainedModel


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = quick_gelu(x)
        x = self.fc2(x)
        return x


class CLIPEncoderLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.mlp = CLIPMLP(n_embd)

    def forward(self, x):
        residue = x
        x = self.layer_norm1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layer_norm2(x)
        x = self.mlp(x)

        return x + residue


class CLIPTextEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_value = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_value
        return x


class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(12, 768) for _ in range(12)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class CLIPTextTransformer(PreTrainedModel, nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPTextEmbedding(49408, 768, 77)
        self.encoder = CLIPEncoder()
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        x = self.embedding(tokens)
        x = self.encoder(x)
        output = self.layernorm(x)
        return output
