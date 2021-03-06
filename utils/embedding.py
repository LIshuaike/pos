# -*- encoding: utf-8 -*-

import torch


class Embedding():
    def __init__(self, tokens, vectors, unk=None):
        super(Embedding, self).__init__()
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, fp, unk=None):
        with open(fp, encoding='utf-8') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                for s in splits])
        embedding = cls(tokens, vectors, unk=unk)

        return embedding