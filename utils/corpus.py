# -*- encoding: utf-8 -*-

from collections import namedtuple

Sentence = namedtuple('Sentence', [
    'ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD',
    'PDEPREL'
],
                      defaults=[None] * 10)


class Corpus():
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __repr__(self):
        return '\n'.join('\n'.join('\t'.join(map(str, i))
                                   for i in zip(*(f for f in sentence if f))) +
                         '\n' for sentence in self)

    @property
    def words(self):
        return [list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        return [list(sentence.CPOS) for sentence in self]

    @classmethod
    def load(cls, fp, columns=range(2)):
        sentences, columns = [], []
        with open(fp, encoding='utf=8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    if columns:
                        sentences.append(Sentence(*columns))
                    columns = []
                else:
                    for i, column in enumerate(line.split('\t')):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)
            if columns:
                sentences.append(Sentence(*columns))
        corpus = cls(sentences)

        return corpus

    def save(self, fp):
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(f"{self}\n")
