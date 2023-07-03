import math
import torch
import torch.nn as nn


class PositionalEncodings(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        # positional encodings
        self.position_encodings = torch.empty(max_seq_len, d_model)

        # compute positional encodings
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                exponent = 2 * (i // 2) / d_model
                if i % 2 == 0:
                    self.position_encodings[pos, i] = math.sin(pos / 10000 ** exponent)
                else:
                    self.position_encodings[pos, i] = math.cos(pos / 10000 ** exponent)

    def forward(self, x):
        # add positional encodings
        x = x + self.position_encodings[:x.size(1), :]
        x = self.dropout(x)
        return x


class Head(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.queries = nn.Linear(d_model, d_model)
        self.keys = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.normal_(self.queries.weight, mean=0, std=1.0)
        nn.init.normal_(self.keys.weight, mean=0, std=1.0)
        nn.init.normal_(self.values.weight, mean=0, std=1.0)

    def forward(self, x):
        # queries, keys, values
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        # scaled dot product attention
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        qk = self.softmax(qk)
        qkv = torch.matmul(qk, v)

        return qkv


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.heads = nn.ModuleList([Head(d_model, dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # init weights
        nn.init.normal_(self.linear.weight, mean=0, std=1.0)

    def forward(self, x):
        # split into heads
        x = torch.cat([head(x) for head in self.heads], dim=-1)

        # combine heads
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.heads = nn.ModuleList([Head(d_model, dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # init weights
        nn.init.normal_(self.linear.weight, mean=0, std=1.0)

    def forward(self, x):
        # mask top right of matrix
        mask = torch.triu(torch.ones_like(x), diagonal=1).bool()
        x = x.masked_fill(mask, 0.0)
        # split into heads
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, y):
        x = x + y
        x = self.layer_norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # init weights
        nn.init.normal_(self.linear1.weight, mean=0, std=1.0)
        nn.init.normal_(self.linear2.weight, mean=0, std=1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Transformer(nn.Module):

    def __init__(self, num_heads, d_model, vocab_size, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position_encodings = PositionalEncodings(d_model, dropout=dropout)

        self.multi_head_attention = MaskedMultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.add_norm1 = AddNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        self.add_norm2 = AddNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

        # init weights
        nn.init.normal_(self.linear.weight, mean=0, std=1.0)

    def forward(self, x):
        x = self.embeddings(x)

        x = self.position_encodings(x)
        copy_x = x.clone()

        x = self.multi_head_attention(x)

        x = self.add_norm1(x, copy_x)

        copy_x = x.clone()

        x = self.feed_forward(x)
        x = self.add_norm2(x, copy_x)
        x = self.linear(x)
        return x
