# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CHAR_LSTM(nn.Module):
    def __init__(self, num_chars, embedding_dim, output_size):
        super(CHAR_LSTM, self).__init__()

        # 字嵌入
        self.embed = nn.Embedding(num_chars, embedding_dim)
        # 字嵌入LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=output_size // 2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        mask = x.ne(0)
        lens = mask.sum(dim=1)
        x = pack_padded_sequence(self.embed(x), lens, True, False)
        x, (hidden, _) = self.lstm(x)
        hidden = torch.cat(torch.unbind(hidden), dim=-1)
        return hidden
