import logging
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn


class LabelModel(nn.Module):
    """Parent class for all generative label models.

    Concrete subclasses should implement at least forward(),
    estimate_label_model(), and get_label_distribution().
    """
    def forward(self, *args):
        """Computes the marginal log-likelihood of a batch of observed
        function outputs provided as input.

        :param args: batch of observed function outputs and related metadata
        :return: 1-d tensor of log-likelihoods, one for each input example
        """
        raise NotImplementedError

    def estimate_label_model(self, *args, config=None):
        """Learns the parameters of the label model from observed
        function outputs.

        Subclasses that implement this method should call _do_estimate_label_model()
        if possible, to provide consistent behavior.

        :param args: observed function outputs and related metadata
        :param config: an instance of LearningConfig. If none, will initialize
                       with default LearningConfig constructor
        """
        raise NotImplementedError

    def get_label_distribution(self, *args):
        """Returns the estimated posterior distribution over true labels given
        observed function outputs.

        :param args: observed function outputs and related metadata
        :return: distribution over true labels. Structure depends on model type
        """
        raise NotImplementedError

    def get_most_probable_labels(self, *args):
        """Returns the most probable true labels given observed function outputs.

        :param args: observed function outputs and related metadata
        :return: 1-d Numpy array of most probable labels
        """
        raise NotImplementedError

    def _do_estimate_label_model(self, batches, config):
        """Internal method for optimizing model parameters.

        :param batches: sequence of inputs to forward(). The sequence must
                        contain tuples, even if forward() takes one
                        argument (besides self)
        :param config: an instance of LearningConfig
        """

        # Sets up optimization hyperparameters
        optimizer = torch.optim.SGD(
            self.parameters(), lr=config.step_size, momentum=config.momentum,
            weight_decay=0)
        if config.step_schedule is not None and config.step_size_mult is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.step_schedule, gamma=config.step_size_mult)
        else:
            scheduler = None

        # Iterates over epochs
        for epoch in range(config.epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, config.epochs))
            if scheduler is not None:
                scheduler.step()

            # Sets model to training mode
            self.train()
            running_loss = 0.0

            # Iterates over training data
            for i_batch, inputs in enumerate(batches):
                optimizer.zero_grad()
                log_likelihood = self(*inputs)
                loss = -1 * torch.mean(log_likelihood)
                loss += self._get_regularization_loss()
                loss.backward()
                optimizer.step()
                running_loss += loss
            epoch_loss = running_loss / len(batches)
            logging.info('Train Loss: %.6f', epoch_loss)

    def _get_regularization_loss(self):
        """Gets the value of the regularization loss for the current values of
        the model's parameters

        :return: regularization loss
        """
        return 0.0


class ClassConditionalLabelModel(LabelModel):
    """
    Abstract parent class for generative label models that assume labeling
    functions are conditionally independent given the true label, and that each
    labeling function is characterized by the following parameters:
        * a propensity, which is the probability that it does not abstain
        * class-conditional accuracies, each of which is the probability that
          the labeling function's output is correct given that the true label
          has a certain value. It is assumed that when a labeling function makes
          a mistake, the label it outputs is chosen uniformly at random
    """
    def __init__(self, num_classes, num_lfs, init_acc, acc_prior):
        """Constructor.

        Initializes label source accuracies argument and propensities uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_lfs: number of labeling functions to model
        :param init_acc: initial estimated labeling function accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling
                          function accuracies toward their initial values
        """
        super().__init__()

        # Converts init_acc to log scale
        init_acc = -1 * np.log(1.0 / init_acc - 1) / 2

        init_param = torch.tensor(
            [[init_acc] * num_classes for _ in range(num_lfs)])
        self.accuracy = nn.Parameter(init_param)
        self.propensity = nn.Parameter(torch.zeros([num_lfs]))

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_lfs
        self.init_acc = init_acc
        self.acc_prior = acc_prior

    def get_accuracies(self):
        """Returns the model's estimated labeling function accuracies
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function correctly outputs
                 the true class label, given that it does not abstain
        """
        acc = self.accuracy.detach().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_propensities(self):
        """Returns the model's estimated labeling function propensities, i.e.,
        the probability that a labeling function does not abstain
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function does not abstain
        """
        prop = self.propensity.detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)

    def _get_labeling_function_likelihoods(self, votes):
        """
        Computes conditional log-likelihood of labeling function votes given
        class as an m x k matrix.

        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.

        :param votes: m x n matrix in {0, ..., k}, where m is the sum of the
                      lengths of the sequences in the batch, n is the number of
                      labeling functions and k is the number of classes
        :return: matrix of dimension m x k, where element is the conditional
                 log-likelihood of votes given class
        """
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        # Initializes conditional log-likelihood of votes as an m x k matrix
        cll = torch.zeros(votes.shape[0], self.num_classes)

        # Initializes normalizing constants
        z_prop = self.propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_lfs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        z_acc = self.accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # Subtracts normalizing constant for propensities from cll
        # (since it applies to all outcomes)
        cll -= torch.sum(z_prop)

        # Loops over votes and classes to compute conditional log-likelihood
        for i, j, v in zip(votes.row, votes.col, votes.data):
            for k in range(self.num_classes):
                if v == (k + 1):
                    logp = self.propensity[j] + self.accuracy[j, k] - z_acc[j, k]
                    cll[i, k] += logp
                elif v != 0:
                    logp = self.propensity[j] - self.accuracy[j, k] - z_acc[j, k]
                    logp -= torch.log(torch.tensor(self.num_classes - 1.0))
                    cll[i, k] += logp

        return cll

    def _get_regularization_loss(self):
        """Computes the regularization loss of the model:
        acc_prior * \|accuracy - init_acc\|_2

        :return: value of regularization loss
        """
        return self.acc_prior * torch.norm(self.accuracy - self.init_acc)


class LearningConfig(object):
    """Container for hyperparameters used by label models during learning"""

    def __init__(self):
        """Initializes all hyperparameters to default values"""
        self.epochs = 5
        self.batch_size = 64
        self.step_size = 0.01
        self.step_schedule = None
        self.step_size_mult = None
        self.momentum = 0.9
        self.random_seed = 0


def init_random(seed):
    """Initializes PyTorch and NumPy random seeds.

    Also sets the CuDNN back end to deterministic.

    :param seed: integer to use as random seed
    """
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info("Random seed: %d", seed)
