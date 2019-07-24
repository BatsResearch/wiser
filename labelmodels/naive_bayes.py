from .label_model import ClassConditionalLabelModel, LearningConfig, init_random
import numpy as np
from scipy import sparse
import torch
from torch import nn


class NaiveBayes(ClassConditionalLabelModel):
    """A generative label model that assumes that all labeling functions are
    conditionally independent given the true class label, i.e., the naive Bayes
    assumption.

    Proposed in: A. P. Dawid and A. M. Skene. Maximum likelihood
    estimation of observer error-rates using the EM algorithm.
    Journal of the Royal Statistical Society C, 28(1):20–28, 1979.

    Proposed for labeling functions in: A. Ratner, C. De Sa, S. Wu, D. Selsam,
    and C. Ré. Data programming: Creating large training sets, quickly. In
    Neural Information Processing Systems, 2016.
    """

    def __init__(self, num_classes, num_lfs, init_acc=.9, acc_prior=0.025,
                 balance_prior=0.025, learn_class_balance=True):
        """Constructor.

        Initializes labeling function accuracies using optional argument and all
        other model parameters uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_lfs: number of labeling functions to model
        :param init_acc: initial estimated labeling function accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling
                          function accuracies toward their initial values
        :param learn_class_balance: whether to estimate the distribution over
                                    target classes (True) or assume to be
                                    uniform (False)
        """
        super().__init__(num_classes, num_lfs, init_acc, acc_prior)
        self.class_balance = nn.Parameter(
            torch.zeros([num_classes]), requires_grad=learn_class_balance)

        self.balance_prior = balance_prior

    def forward(self, votes):
        """Computes log likelihood of labeling function outputs for each
        example in the batch.

        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding row in labels
        """
        class_ll = self._get_norm_class_balance()
        conditional_ll = self._get_labeling_function_likelihoods(votes)
        joint_ll = conditional_ll + class_ll
        return torch.logsumexp(joint_ll, dim=1)

    def estimate_label_model(self, votes, config=None):
        """Estimates the parameters of the label model based on observed
        labeling function outputs.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        # Converts to CSR to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)

        batches = self._create_minibatches(
            votes, config.batch_size, shuffle_rows=True)
        self._do_estimate_label_model(batches, config)

    def get_label_distribution(self, votes):
        """Returns the posterior distribution over true labels given labeling
        function outputs according to the model

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: m x k matrix, where each row is the posterior distribution over
                 the true class label for the corresponding example
        """
        # Converts to CSR to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)

        labels = np.ndarray((votes.shape[0], self.num_classes))
        batches = self._create_minibatches(votes, 4096, shuffle_rows=False)

        offset = 0
        for votes, in batches:
            class_balance = self._get_norm_class_balance()
            lf_likelihood = self._get_labeling_function_likelihoods(votes)
            jll = class_balance + lf_likelihood
            for i in range(votes.shape[0]):
                p = torch.exp(jll[i, :] - torch.max(jll[i, :]))
                p = p / p.sum()
                for j in range(self.num_classes):
                    labels[offset + i, j] = p[j]
            offset += votes.shape[0]

        return labels

    def get_most_probable_labels(self, votes):
        """Returns the most probable true labels given observed function outputs.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d Numpy array of most probable labels
        """
        return np.argmax(self.get_label_distribution(votes), axis=1) + 1

    def get_class_balance(self):
        """Returns the model's estimated class balance

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        return np.exp(self._get_norm_class_balance().detach().numpy())

    def _create_minibatches(self, votes, batch_size, shuffle_rows=False):
        if shuffle_rows:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :]

        # Creates minibatches
        batches = [(sparse.coo_matrix(
            votes[i * batch_size: (i + 1) * batch_size, :],
            copy=True),)
            for i in range(int(np.ceil(votes.shape[0] / batch_size)))
        ]

        return batches

    def _get_regularization_loss(self):
        neg_entropy = 0.0
        norm_class_balance = self._get_norm_class_balance()
        exp_class_balance = torch.exp(norm_class_balance)
        for k in range(self.num_classes):
            neg_entropy += norm_class_balance[k] * exp_class_balance[k]
        entropy_prior = self.balance_prior * neg_entropy

        return super()._get_regularization_loss() + entropy_prior

    def _get_norm_class_balance(self):
        return self.class_balance - torch.logsumexp(self.class_balance, dim=0)
