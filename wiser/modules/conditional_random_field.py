import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField

class WiserConditionalRandomField(ConditionalRandomField):
    def expected_log_likelihood(
            self,
            logits: torch.Tensor,
            mask: torch.ByteTensor,
            unary_marginals: torch.FloatTensor,
            pairwise_marginals: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the expected log likelihood of CRFs defined by batch of logits
        with respect to a reference distribution over the random variables.

        Parameters
        ----------
        logits : torch.Tensor, required
            The logits that define the CRF distribution, of shape
            ``(batch_size, seq_len, num_tags)``.
        unary_marginals : torch.Tensor, required
            Marginal probability that each sequence element is a particular tag,
            according to the reference distribution. Shape is
            ``(batch_size, seq_len, num_tags)``.
        pairwise_marginals : torch.Tensor, optional (default = ``None``)
            Marginal probability that each pair of sequence elements is a
            particular pair of tags, according to the reference distribution.
            Shape is ``(batch_size, seq_len - 1, num_tags, num_tags)``, so
            pairwise_marginals[:, 0, 0, 0] is the probability that the first
            and second tags in each sequence are both 0,
            pairwise_marginals[:, 1, 0, 0] is the probability that the second
            and and third tags in each sequence are both 0, etc. If None,
            pairwise_marginals will be computed from unary_marginals assuming
            that they are independent in the reference distribution.
        mask : ``torch.ByteTensor``
            The text field mask for the input tokens of shape
            ``(batch_size, seq_len)``.
        """
        batch_size, seq_len, num_tags = logits.size()

        # We compute the partition function before rearranging the inputs
        partition = self._input_likelihood(logits, mask)
        # Transpose batch size and sequence dimensions
        logits = logits.transpose(0, 1).contiguous()                 # (seq_len, batch_size, num_tags)
        mask = mask.float().transpose(0, 1).contiguous()             # (seq_len, batch_size)

        unary_marginals = unary_marginals.transpose(0, 1)            # (seq_len, batch_size, num_tags)
        unary_marginals = unary_marginals.contiguous()
        if pairwise_marginals is not None:
            pairwise_marginals = pairwise_marginals.transpose(0, 1)  # (seq_len - 1, batch_size, num_tags, num_tags)
            pairwise_marginals = pairwise_marginals.contiguous()
        else:
            pairwise_marginals = torch.zeros(
                                    (seq_len - 1, batch_size, num_tags, num_tags),
                                    device=logits.device.type)

            for i in range(seq_len - 1):
                for j in range(batch_size):
                    temp1 = unary_marginals[i, j]
                    temp2 = unary_marginals[i+1, j]
                    temp = torch.ger(temp1, temp2)
                    pairwise_marginals[i, j, :, :] = temp


        # Start with the transition scores from start_tag to the
        # first tag in each input
        if self.include_start_end_transitions:
            temp = self.start_transitions.unsqueeze(0)               # (1, num_tags)
            temp = temp * unary_marginals[0]                         # (batch_size, num_tags)
            score = temp.sum(dim=1)                                  # (batch_size,)
        else:
            score = torch.zeros((batch_size,), device=logits.device.type) # (batch_size,)


        # Add up the scores for the expected transitions and all
        # the inputs but the last
        for i in range(seq_len - 1):
            # Adds contributions from logits
            temp = logits[i] * unary_marginals[i]                    # (batch_size, num_tags)
            temp = temp.sum(dim=1)                                   # (batch_size,)
            score += temp * mask[i]

            # Adds contributions from transitions from i to i+1
            temp = self.transitions.unsqueeze(0)                     # (1, num_tags, num_tags)
            temp = temp * pairwise_marginals[i]                      # (batch_size, num_tags, num_tags)
            temp = temp.sum(dim=2).sum(dim=1)                        # (batch_size,)
            score += temp * mask[i+1]

        # Transition from last state to "stop" state.
        # Computes score of transitioning to `stop_tag` from
        # each last token.
        if self.include_start_end_transitions:
            # To start, we need to find the last token for
            # each instance.
            index0 = mask.sum(dim=0).long() - 1                      # (batch_size,)
            index1 = torch.arange(0, batch_size, dtype=torch.long)   # (batch_size,)
            last_marginals = unary_marginals[index0, index1, :]      # (batch_size, num_tags)

            temp = self.end_transitions.unsqueeze(0)                 # (1, num_tags)
            temp = temp * last_marginals                             # (batch_size, num_tags)
            temp = temp.sum(dim=1)                                   # (batch_size,)
            score += temp

        # Adds the last input if it's not masked.
        last_scores = logits[-1] * unary_marginals[-1]               # (batch_size, num_tags)
        last_scores = last_scores.sum(dim=1)                         # (batch_size,)
        score += last_scores * mask[-1]

        # Finally we subtract partition function and return sum
        return torch.sum(score - partition)
