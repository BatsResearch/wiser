import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField


class WiserConditionalRandomField(ConditionalRandomField):

    def construct_pairwise_marginals(self, unary_marginals):
        """
       Given a tensor of unary marginals and a mask, returns the pairwise_marginals by multiplying adjacent
       unary marginals. This method assumes independence between adjacent terms, similar to naive Bayes.

       Parameters
       ----------
       unary_marginals : ``tensor``, required
           The (batch_size, seq_len, num_tags, num_tags) tensor of unary marginals.

       Returns
       -------
           A torch.Tensor of size (batch_size, seq_len, num_tags, num_tags) containing
           the pariwise marginals derived from the unary marginals.
       """

        pairwise_marginals = torch.einsum('ijk, ijl -> ijkl',
                                          unary_marginals[:, :-1, :], unary_marginals[:, 1:, :])
        return torch.cat((pairwise_marginals, torch.zeros(64, 1, 5, 5)), 1)

    def batch_start_transitions(self, unary_marginals):
        """
        Given a tensor of unary marginals, returns the batch start transitions.

        Parameters
        ----------
        unary_marginals : ``tensor``, required
            The (batch_size, seq_len, num_tags) tensor of unary marginals.

        Returns
        -------
            A torch.Tensor of size (batch_size,) containing end transitions for each batch.
        """
        return torch.mv(unary_marginals[:, 0, :], self.start_transitions)

    def batch_end_transitions(self, unary_marginals, mask):
        """
        Given a tensor of unary marginals and a mask, returns the batch end transitions

        Parameters
        ----------
        unary_marginals : ``tensor``, required
            The (batch_size, seq_len, num_tags) tensor of unary marginals.

        mask : ``tensor``, required
            The (batch_size, seq_len) mask of tokens

        Returns
        -------
            A torch.Tensor of size (batch_size,) containing end transitions for each batch
        """
        batch_size = unary_marginals.size(0)
        ix0 = torch.arange(0, batch_size, dtype=torch.long)
        ix1 = mask.sum(-1).long() - 1
        return torch.mv(unary_marginals[ix0, ix1, :], self.end_transitions)

    def batch_unary_marginals(self, logits, unary_marginals, mask):
        """
       Given a tensor of unary marginals, the logits, and a mask, returns the batch unary_marginals.

       Parameters
       ----------
       logits : ``tensor``, required
            The (batch_size, seq_len, num_tags) tensor containing the logits.
       unary_marginals : ``tensor``, required
           The (batch_size, seq_len, num_tags) tensor of unary marginals.
       mask : ``tensor``, required
           The (batch_size, seq_len) mask of tokens.
       Returns
       -------
           A torch.Tensor of size (batch_size,) containing the unary marginals for each batch.
       """
        return torch.einsum('ij, ij -> i',
                            (unary_marginals * logits).sum(-1), mask)

    def batch_pairwise_marginals(self, pairwise_marginals, mask):
        """
        Given a tensor of pairwise marginals and a mask, returns the batch pairwise_marginals.

        Parameters
        ----------
        pairwise_marginals : ``tensor``, required
            The (batch_size, seq_len, num_tags, num_tags) tensor of pairwise marginals.
        mask : ``tensor``, required
            The (batch_size, seq_len) mask of tokens.
        Returns
        -------
            A torch.Tensor of size (batch_size,) containing end transitions for each batch.
        """
        return (torch.einsum('ijkl, kl -> ij', pairwise_marginals,
                             self.transitions)[:, :-1] * mask[:, 1:]).sum(-1)

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

        mask = mask.float()

        if pairwise_marginals is None:
            pairwise_marginals = self.construct_pairwise_marginals(
                unary_marginals)

        score = self.batch_start_transitions(unary_marginals)
        score += self.batch_unary_marginals(logits, unary_marginals, mask)
        score += self.batch_pairwise_marginals(pairwise_marginals, mask)
        score += self.batch_end_transitions(unary_marginals, mask)

        partition = self._input_likelihood(logits, mask)

        # Subtracts partition function and returns sum
        return torch.sum(score - partition)
