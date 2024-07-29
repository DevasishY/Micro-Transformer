import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Input embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embed(x) * np.sqrt(self.d_model)


# positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # useful when we saving the model.

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return x


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


# feed forward network
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.model(x)


# skip connection for vanishing gradinet problem, to transfer strong signal.
class ResidualConnection(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return self.norm(x + sublayer(x))


# Attention block
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv

    def forward(self, q, k, v, flag):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        if flag == "Encoder":  # checking encoder or decoder
            print("In encoder")
            attention_value_matrix = F.scaled_dot_product_attention(
                Q, K, V, is_causal=False
            )

        elif flag == "Decoder":
            attention_value_matrix = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True
            )
            print("In decoder")
        else:
            print("Error")
        return attention_value_matrix
