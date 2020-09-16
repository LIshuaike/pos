import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from module import CRF, CHAR_LSTM


class CHAR_LSTM_CRF(nn.Module):
    def __init__(self,
                 num_chars,
                 char_embedding_dim,
                 char_output_size,
                 num_words,
                 embedding_dim,
                 hidden_size,
                 output_size,
                 dropratio=0.5):
        super(CHAR_LSTM_CRF, self).__init__()

        self.embed = nn.Embedding(num_words, embedding_dim)
        # 字符嵌入LSTM层
        self.char_lstm = CHAR_LSTM(num_chars, char_embedding_dim,
                                   char_output_size)
        # 词嵌入LSTM层
        self.word_lstm = nn.LSTM(input_size=embedding_dim + char_output_size,
                                 hidden_size=hidden_size,
                                 num_layers=2,
                                 batch_first=True,
                                 bidirectional=True)

        # 输出层
        self.out = nn.Linear(hidden_size * 2, output_size)

        self.crf = CRF(output_size, batch_first=True)

        self.drop = nn.Dropout(dropratio)

    def load_pretrained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, False)

    def forward(self, x, char_x):
        # B, T = x.shape
        mask = x.ne(0)
        lens = mask.sum(dim=1)
        # 获取词嵌入向量
        x = self.embed(x[mask])
        char_x = self.char_lstm(char_x[mask])
        x = torch.cat((x, char_x), dim=-1)
        x = self.drop(x)

        # x = pack_padded_sequence(x, lens, True)
        x = pack_sequence(torch.split(x, lens.tolist()), False)
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)

        return self.out(x)