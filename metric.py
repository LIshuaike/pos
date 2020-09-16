import torch


class Metric(object):
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class AccMetric(Metric):
    def __init__(self, eps=1e-5):
        super(AccMetric, self).__init__()

        self.tp = 0.0
        self.total = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            self.tp += torch.sum(pred == gold).item()
            self.total += len(gold)

    def __repr__(self):
        return f"Accuracy: {self.score:.2%}"

    @property
    def score(self):
        acc = self.tp / (self.total + self.eps)
        return acc
