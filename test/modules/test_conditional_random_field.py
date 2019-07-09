import math
import numpy as np
import torch
from torch import optim
import unittest
from wiser.modules import WiserConditionalRandomField


class TestWiserConditionalRandomField(unittest.TestCase):
    def setUp(self):
        """
        Sets all random seeds before each test.
        """
        seed = 0
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def test_distribution_learning_matches(self):
        """
        Trains two CRFs on the same data, once represented as tags and once
        represented as a distribution, in order to test learning via maximizing
        expected_log_likelihood. Test passes iff learned parameters of two CRFs
        match.
        """
        num_labels = 3
        num_examples = 100
        min_seq_len = 8
        max_seq_len = 12
        logits = torch.Tensor([-0.5, -.25, 0])
        transitions = torch.eye(3)
        start = torch.Tensor([0.25, 0, 0.25])
        start -= torch.max(start)
        end = start

        tags, mask = _generate_data(
            num_examples, min_seq_len, max_seq_len, logits, transitions,
            start, end)

        # Trains first CRF on tags
        crf1 = LinearCrf(num_labels)
        # Initializes all weights to zero so we can compare with other CRF
        torch.nn.init.zeros_(crf1.linear)
        torch.nn.init.zeros_(crf1.transitions)
        torch.nn.init.zeros_(crf1.start_transitions)
        torch.nn.init.zeros_(crf1.end_transitions)
        _train_crf_tags(crf1, tags, mask, 50, 32)

        # Converts tags to exact distribution
        distribution = torch.zeros(tags.shape[0], tags.shape[1], num_labels)
        for i in range(tags.shape[0]):
            for j in range(tags.shape[1]):
                distribution[i, j, tags[i, j]] = 1.0

        # Trains second CRF on distribution
        crf2 = LinearCrf(num_labels)
        torch.nn.init.zeros_(crf2.linear)
        torch.nn.init.zeros_(crf2.transitions)
        torch.nn.init.zeros_(crf2.start_transitions)
        torch.nn.init.zeros_(crf2.end_transitions)
        _train_crf_distribution(crf2, distribution, mask, 50, 32)

        # Tests that all parameters match
        self.assertLess(torch.norm(crf1.linear - crf2.linear), 1e-3)
        self.assertLess(torch.norm(crf1.transitions - crf2.transitions), 1e-3)
        self.assertLess(torch.norm(crf1.start_transitions - crf2.start_transitions), 1e-3)
        self.assertLess(torch.norm(crf1.end_transitions - crf2.end_transitions), 1e-3)


class LinearCrf(WiserConditionalRandomField):
    """Wraps WiserConditionalRandomField to learn fixed logits for each
    sequence element."""
    def __init__(self, num_tags):
        super().__init__(num_tags, None, True)
        self.linear = torch.nn.Parameter(torch.Tensor(num_tags))
        torch.nn.init.normal_(self.linear)

    def forward(self,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return super().forward(
            self.linear.repeat(tags.shape[0], tags.shape[1], 1),
            tags,
            mask
        )

    def expected_log_likelihood(
            self,
            distribution: torch.Tensor,
            mask: torch.ByteTensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return super().expected_log_likelihood(
            self.linear.repeat(distribution.shape[0], distribution.shape[1], 1),
            mask,
            distribution
        )


def _train_crf_tags(crf, tags, mask, epochs, batch_size):
    num_batches = math.ceil(tags.shape[0] / batch_size)
    optimizer = optim.Adam(crf.parameters())
    for _ in range(epochs):
        for i in range(num_batches):
            batch_tags = tags[i * batch_size:(i+1) * batch_size]
            batch_mask = mask[i * batch_size:(i+1) * batch_size]
            crf.zero_grad()
            loss = -crf(batch_tags, batch_mask)
            loss.backward()
            optimizer.step()


def _train_crf_distribution(crf, distribution, mask, epochs, batch_size):
    num_batches = math.ceil(distribution.shape[0] / batch_size)
    optimizer = optim.Adam(crf.parameters())
    for _ in range(epochs):
        for i in range(num_batches):
            batch_distribution = distribution[i * batch_size:(i + 1) * batch_size]
            batch_mask = mask[i * batch_size:(i + 1) * batch_size]
            crf.zero_grad()
            loss = -crf.expected_log_likelihood(batch_distribution, batch_mask)
            loss.backward()
            optimizer.step()


def _generate_data(num_examples, min_seq_len, max_seq_len, logits, transitions,
                   start, end):
    tags = torch.zeros((num_examples, max_seq_len), dtype=torch.long)
    mask = torch.zeros((num_examples, max_seq_len), dtype=torch.long)

    for i in range(num_examples):
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        seq = _generate_seq(seq_len, logits, transitions, start, end)
        for j in range(seq_len):
            tags[i, j] = seq[j]
            mask[i, j] = 1

    return tags, mask


def _generate_seq(seq_len, logits, transitions, start, end, gibbs_rounds=5):
    seq = torch.zeros((seq_len,), dtype=torch.long)

    # Randomly initializes the sequence
    for i in range(seq_len):
        seq[i] = np.random.randint(logits.shape[0])

    # Performs rounds of Gibbs sampling
    p = torch.zeros((logits.shape[0],))
    for _ in range(gibbs_rounds):
        for i in range(seq_len):
            if i == 0:
                # Neighbor only on right
                p[:] = logits
                p += transitions[:, seq[i+1]]
                p += start
            elif i == seq_len - 1:
                # Neighbor only on left
                p[:] = logits
                p += transitions[seq[i-1], :]
                p += end
            else:
                # Neighbors on both sides
                p[:] = logits
                p += transitions[seq[i-1], :]
                p += transitions[:, seq[i + 1]]

            p = torch.exp(p - torch.max(p))
            p = p / torch.sum(p)
            seq[i] = float(np.argmax(np.random.multinomial(1, p)))

#    print(seq)
    return seq


if __name__ == '__main__':
    unittest.main()
