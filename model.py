import math
import torch
import torch.nn as nn


class PositionalEncodings(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=0.1)

        # positional encodings
        self.position_encodings = torch.empty(max_seq_len, d_model)

        # compute positional encodings
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                exponent = 2 * (i // 2) / d_model
                self.position_encodings[i, pos] = math.sin(pos / 10000 ** exponent)
                self.position_encodings[i + 1, pos] = math.cos(pos / 10000 ** exponent)

    def forward(self, x):
        # add positional encodings
        x = x + self.position_encodings[:, :x.size(1)]
        return self.dropout(x)


class Head(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.queries = nn.Linear(d_model, d_model)
        self.keys = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # queries, keys, values
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        x = torch.matmul(scores, v)
        return x