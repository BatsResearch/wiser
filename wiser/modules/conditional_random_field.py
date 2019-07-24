import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField
import pdb

class WiserConditionalRandomField(ConditionalRandomField):
    def expected_log_likelihood(
            self,
            logits: torch.Tensor,
            mask: torch.ByteTensor,
            unary_marginals: torch.FloatTensor,
            pairwise_marginals: torch.Tensor = None,
            vote_mask: torch.Tensor = None) -> torch.Tensor:
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

        if pairwise_marginals is None:
            pairwise_marginals = torch.zeros(
                                    (batch_size, seq_len - 1, num_tags, num_tags),
                                    device=logits.device.type)

            for i in range(batch_size):
                for j in range(seq_len - 1):
                    temp = torch.ger(unary_marginals[i, j],  unary_marginals[i, j+1])
                    pairwise_marginals[i, j, :, :] = temp

        score = torch.zeros((batch_size), device=logits.device.type) # (batch_size,)

        score += ((unary_marginals * logits).sum(-1) * mask.float()).sum(-1)
        score += (torch.einsum('abcd, cd -> ab', pairwise_marginals, self.transitions) * mask[:,1:].float()).sum(-1)

        if self.include_start_end_transitions:

            # Start with the transition scores from start_tag to the
            # first tag in each input
            start_transitions = self.start_transitions.unsqueeze(1)             # (batch_size, 1)
            score += torch.mm(unary_marginals[:, 0], start_transitions).view(-1)

            # Computes score of transitioning to `stop_tag` from
            # each last token.
            index0 = torch.arange(0, batch_size, dtype=torch.long)   # (batch_size,)
            index1 = mask.sum(dim=1).long() - 1                      # (batch_size,)

            end_transitions = self.end_transitions.unsqueeze(1)      # (batch_size, 1)
            score += torch.mm(unary_marginals[index0, index1, :], end_transitions).view(-1)

        # Finally we subtract partition function and return sum
        return torch.sum(score - partition)
