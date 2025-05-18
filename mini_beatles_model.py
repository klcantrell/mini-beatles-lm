import math
import torch
import torch.nn as nn

# Hyperparameters (shared, now with flexible model size)
default_embed_dim   = 256
default_n_heads     = 8
default_n_layers    = 8

default_ff_dim      = default_embed_dim * 4
default_max_len     = 128

default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def make_causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class MiniBeatlesLM(nn.Module):
    def __init__(self, vocab_size, pad_token_id, embed_dim=default_embed_dim, n_heads=default_n_heads, n_layers=default_n_layers, ff_dim=None, max_len=default_max_len):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
        self.tok_emb   = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = PositionalEncoding(embed_dim, max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids):
        B, T = input_ids.size()
        pad_mask = input_ids == self.pad_token_id
        x = self.tok_emb(input_ids)
        x = self.pos_emb(x)
        tgt_mask = make_causal_mask(T, x.device)
        x = self.decoder(
            tgt=x,
            memory=x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
        )
        logits = self.lm_head(x)
        return logits

# For import convenience
embed_dim = default_embed_dim
n_heads = default_n_heads
n_layers = default_n_layers
ff_dim = default_ff_dim
max_len = default_max_len
