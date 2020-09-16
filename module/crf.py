import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """
    def __init__(self, num_tags, batch_first=False):
        super(CRF, self).__init__()

        self.n_tag = num_tags
        self.batch_first = batch_first

        self.start_transtions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transtions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None):
        batch_size = emissions.size(1)
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
         Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        # shape (batch_size, )
        log_numerator = self._score(emissions, tags, mask)
        # shape(batch_size, )
        log_denominator = self._forward_alg(emissions, mask)

        return (log_denominator - log_numerator) / batch_size

    def _score(self, emissions, tags, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        # Start transition score and first emission
        # score = self.start_transtions[tags[0]]
        # score += emissions[0, torch.arange(batch_size), tags[0]]

        # for i in range(1, seq_length):
        #     # Transition score to next tag, only added if next timestep is valid (mask == 1)
        #     # shape: (batch_size,)
        #     score += self.transtions[tags[i - 1], tags[i]] * mask[i]

        #     # Emission score for next tag, only added if next timestep is valid (mask == 1)
        #     # shape: (batch_size,)
        #     score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape:(batch_size, )
        # seq_ends = mask.sum(dim=0) - 1
        # # shape:(batch_size, )
        # last_tags = tags[seq_ends, torch.arange(batch_size)]
        # score += self.end_transitions[last_tags]
        scores = torch.zeros_like(
            tags,
            dtype=torch.float)  #[sen_len, batch_size, labels_num] 必须指定为float

        # 加上句间迁移分数
        scores[1:] += self.transitions[tags[:-1], tags[1:]]
        # 加上发射分数
        scores += emissions.gather(dim=2, index=tags.unsqueeze(2)).squeeze(2)
        # 通过掩码过滤分数
        score = scores.masked_select(mask).sum()

        # 获取序列最后的词性的索引
        ends = mask.sum(dim=0).view(1, -1) - 1
        # 加上句首迁移分数
        score += self.start_transtions[tags[0]].sum()
        # 加上句尾迁移分数
        score += self.end_transitions[tags.gather(dim=0, index=ends)].sum()

        return score

    def _forward_alg(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # # Start transition score and first emission; score has size of
        # # (batch_size, num_tags) where for each batch, the j-th column stores
        # # the score that the first timestep has tag j
        # # shape: (batch_size, num_tags)
        # score = self.start_transtions + emissions[0]

        # for i in range(1, seq_length):
        #     # Broadcast score for every possible next tag
        #     # shape: (batch_size, num_tags, 1)
        #     broadcast_score = score.unsqueeze(2)

        #     # Broadcast emission score for every possible current tag
        #     # shape: (batch_size, 1, num_tags)
        #     broadcast_emissions = emissions[i].unsqueeze(1)

        #     # Compute the score tensor of size (batch_size, num_tags, num_tags) Whethe
        #     # for each sample, entry at row i and column j stores the sum of scores of all
        #     # possible tag sequences so far that end with transitioning from tag i to tag j
        #     # and emitting
        #     # shape: (batch_size, num_tags, num_tags)
        #     next_score = broadcast_score + self.transitions.unsqueeze + broadcast_emissions

        #     # Sum over all possible current tags, but we're in score space, so a sum
        #     # becomes a log-sum-up: for each sample, entty i stores the sum of scores of
        #     # all possible tag sequences so far, that end in tag i
        #     # shape:(batch_size, num_tags)
        #     next_score = torch.logsumexp(next_score, dim=1)

        #     # Set score to the next score if this timestep is valid(mask==1)
        #     # shape:(batch_size, num_tags)
        #     score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # # End transition score
        # # shape:(batch_size, num_tags)
        # score += self.end_transitions

        # # Sum (log-sum-exp) over all possible tags
        # # shape:(batch_size,)
        # return torch.logsumexp(score, dim=1).sum()
        alpha = self.start_transtions + emissions[0]

        for i in range(1, seq_length):
            trans_i = self.transitions.unsqueeze(0)
            emit_i = emissions[i].unsqueeze(1)
            mask_i = mask[i].unsqueeze(1).expand_as(alpha)
            scores = trans_i + emit_i + alpha.unsqueeze(2)
            scores = torch.logsumexp(scores, dim=1)
            alpha[mask_i] = scores[mask_i]

        return torch.logsumexp(alpha + self.end_transitions, dim=1).sum()

    def viterbi(self, emissions, mask):
        if self.batch_first:
            emissions, mask = emissions.transpose(0, 1), mask.t()
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        seq_length, batch_size = mask.shape
        device = emissions.device

        # # Start transition and frist emission
        # # - score is a tensor of size (batch_size, num_tags) where for every batch,
        # #   value at column j stores the score of the best tag sequence so far that ends
        # #   with tag j
        # # shape:(batch_size, num_tags)
        # score = self.start_transtions + emissions[0]

        # # - history_idx saves where the best tags candidate transitioned from; this is used
        # #   when we trace back the best tag sequence
        # # - oor_idx saves the best tags candidate transitioned from at the positions
        # #   where mask is 0, i.e. out of range (oor)
        # history_idx = torch.zeros_like(emissions)
        # oor_idx = torch.zeros_like(emissions)
        # oor_tag = torch.zeros_like(mask)

        # # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # # for every possible next tag
        # for i in range(1, seq_length):
        #     # Broadcast viterbi score for every possible next tag
        #     # shape: (batch_size, num_tags, 1)
        #     broadcast_score = score.unsqueeze(2)

        #     # Broadcast emission score for every possible current tag
        #     # shape: (batch_size, 1, num_tags)
        #     broadcast_emission = emissions[i].unsqueeze(1)

        #     # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        #     # for each sample, entry at row i and column j stores the score of the best
        #     # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        #     # shape: (batch_size, num_tags, num_tags)
        #     next_score = broadcast_score + self.transitions + broadcast_emission

        #     # Find the maximum score over all possible current tag
        #     # shape: (batch_size, num_tags)
        #     next_score, indices = next_score.max(dim=1)

        #     # Set score to the next score if this timestep is valid (mask == 1)
        #     # and save the index that produces the next score
        #     # shape: (batch_size, num_tags)
        #     score = torch.where(mask[i].unsqueeze(-1), next_score, score)
        #     indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
        #     history_idx[i - 1] = indices

        # # End transition score
        # # shape: (batch_size, num_tags)
        # end_score = score + self.end_transitions
        # _, end_tag = end_score.max(dim=1)

        # # shape: (batch_size,)
        # seq_ends = mask.long().sum(dim=0) - 1

        # # insert the best tag at each sequence end (last position with mask == 1)
        # history_idx = history_idx.transpose(1, 0).contiguous()
        # history_idx.scatter_(
        #     1,
        #     seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
        #     end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        # history_idx = history_idx.transpose(1, 0).contiguous()

        # # The most probable path for each sequence
        # best_tags_arr = torch.zeros((seq_length, batch_size),
        #                             dtype=torch.long,
        #                             device=device)
        # best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # for idx in range(seq_length - 1, -1, -1):
        #     best_tags = torch.gather(history_idx[idx], 1, best_tags)
        #     best_tags_arr[idx] = best_tags.data.view(batch_size)

        # return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

        # # 解码
        # predicts = []

        # return torch.cat(predicts)

        lens = mask.sum(dim=0)
        delta = torch.zeros_like(emissions)
        paths = torch.zeros_like(emissions, dtype=torch.long)  #必须指定为long

        delta[0] = self.start_transtions + emissions[0]

        for i in range(1, seq_length):
            #shape:(batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)
            # shape"(batch_size, num_tags, num_tags)
            scores = self.transitions + broadcast_emission + delta[
                i - 1].unsqueeze(2)
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            previous = torch.argmax(delta[length - 1, i] +
                                    self.end_transitions)
            predict = [previous]
            for j in reversed(range(1, length)):
                previous = paths[j, i, previous]
                predict.append(previous)
            # 反转预测序列并保存
            predicts.append(torch.tensor(predict, device=device).flip(0))

        # return torch.cat(predicts)
        return predicts
