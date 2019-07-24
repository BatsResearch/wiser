from .label_model import ClassConditionalLabelModel, LearningConfig, init_random
import numpy as np
from scipy import sparse
import torch
from torch import nn


class LinkedHMM(ClassConditionalLabelModel):
    """A generative label model that treats a sequence of true class labels as a
    Markov chain, as in a hidden Markov model, and treats all labeling functions
    as conditionally independent given the corresponding true class label, as
    in a Naive Bayes model.

    In addition, the linked HMM captures linking functions that are conditioned
    on consecutive pairs of hidden states, with distributions that depend on
    whether the linked states are the same or different. The outputs of these
    functions are represented as a separate m x n link matrix in {-1, 0, 1},
    where m is the sum of the lengths of the sequences in the data and n is the
    number of linking functions. An output of 1 at entry i, j indicates that
    true labels i-1 and i have the same value, -1 indicates they do not, and
    0 means the linking function abstains.

    The link matrix always has a row of 0 corresponding to the first element in
    each sequence in the data, because linking functions operate on pairs of
    elements. This does not count as an abstention.
    """

    def __init__(self, num_classes, num_labeling_funcs, num_linking_funcs,
                 init_acc=.9, acc_prior=1, balance_prior=1):
        """Constructor.

        Initializes labeling and linking function accuracies using optional
        argument and all other model parameters uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_labeling_funcs: number of labeling functions to model
        :param num_linking_funcs: number of linking functions to model
        :param init_acc: initial estimated labeling and linking function
                         accuracy, must be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling and
                          linking function accuracies toward their initial values
        """
        super().__init__(num_classes, num_labeling_funcs, init_acc, acc_prior)

        self.link_accuracy = nn.Parameter(
            torch.tensor([self.init_acc] * num_linking_funcs))
        self.link_propensity = nn.Parameter(torch.zeros([num_linking_funcs]))
        self.start_balance = nn.Parameter(torch.zeros([num_classes]))
        self.transitions = nn.Parameter(torch.zeros([num_classes, num_classes]))

        # Saves state
        self.num_linking_funcs = num_linking_funcs
        self.balance_prior = balance_prior

    def forward(self, label_votes, link_votes, seq_starts):
        """
        Computes log likelihood of sequence of labeling and linking function
        outputs for each (sequence) example in batch.

        For efficiency, this function prefers that label_votes and link_votes
        are instances of scipy.sparse.coo_matrix. You can avoid a conversion by
        passing them in as this class.

        :param label_votes: m x n matrix in {0, ..., k}, where m is the sum of
                            the lengths of the sequences in the batch, n is the
                            number of labeling functions and k is the number of
                            classes
        :param link_votes: m x n matrix in {-1, 0, 1}, where m is the sum of
                           the lengths of the sequences in the batch and n is the
                           number of linking functions
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, label_votes[seq_starts[i]]
                           is the row vector of labeling function outputs for the
                           first element in the ith sequence
        :return: vector of length l, where element is the log-likelihood of the
                 corresponding sequence of outputs in votes
        """
        jll = self._get_labeling_function_likelihoods(label_votes)
        link_cll = self._get_linking_function_likelihoods(link_votes)
        norm_start_balance = self._get_norm_start_balance()
        norm_transitions = self._get_norm_transitions()
        for i in range(0, jll.shape[0]):
            if i in seq_starts:
                jll[i] += norm_start_balance
            else:
                joint_class_pair = jll[i-1, :].clone().unsqueeze(1)
                joint_class_pair = joint_class_pair.repeat(1, self.num_classes)
                joint_class_pair += norm_transitions

                # Adds contributions from links
                joint_class_pair += link_cll[i]

                # Finishes computing joint log likelihood
                jll[i] += joint_class_pair.logsumexp(0)
        seq_ends = [x - 1 for x in seq_starts] + [jll.shape[0]-1]
        seq_ends.remove(-1)
        mll = torch.logsumexp(jll[seq_ends], dim=1)
        return mll

    def estimate_label_model(self, label_votes, link_votes, seq_starts, config=None):
        """Estimates the parameters of the label model based on observed
        labeling and linking function outputs.

        Note that a minibatch's size refers to the number of sequences in the
        minibatch.

        :param label_votes: m x n matrix in {0, ..., k}, where m is the sum of
                            the lengths of the sequences in the batch, n is the
                            number of labeling functions and k is the number of
                            classes
        :param link_votes: m x n matrix in {-1, 0, 1}, where m is the sum of
                           the lengths of the sequences in the batch and n is the
                           number of linking functions
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, label_votes[seq_starts[i]]
                           is the row vector of labeling function outputs for the
                           first element in the ith sequence
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        # Converts to CSR and integers to standardize input
        label_votes = sparse.csr_matrix(label_votes, dtype=np.int)
        link_votes = sparse.csr_matrix(link_votes, dtype=np.int)
        seq_starts = np.array(seq_starts, dtype=np.int)

        batches = self._create_minibatches(
            label_votes, link_votes, seq_starts, config.batch_size, shuffle_seqs=True)

        self._do_estimate_label_model(batches, config)

    def get_label_accuracies(self):
        """Alias for ClassConditionalModel.get_accuracies()
        :return: estimated labeling function accuracies
        """
        return self.get_accuracies()

    def get_link_accuracies(self):
        """Returns the model's estimated linking function accuracies
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding linking function correctly identifies a pair
                 of consecutive labels, given that it does not abstain
        """
        acc = self.link_accuracy.detach().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_label_propensities(self):
        """Alias for ClassConditionalModel.get_propensities()
        :return: estimated labeling function propensities
        """
        return self.get_propensities()

    def get_link_propensities(self):
        """Returns the model's estimated linking function propensities, i.e.,
        the probability that a linking function does not abstain
        :return: a NumPy array with one element in [0,1] for each linking
                 function, representing the estimated probability that
                 the corresponding linking function does not abstain
        """
        prop = self.link_propensity.detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)

    def get_most_probable_labels(self, label_votes, link_votes, seq_starts):
        """
        Computes the most probable underlying sequence nodes given function
        outputs

        :param label_votes: m x n matrix in {0, ..., k}, where m is the sum of
                            the lengths of the sequences in the batch, n is the
                            number of labeling functions and k is the number of
                            classes
        :param link_votes: m x n matrix in {-1, 0, 1}, where m is the sum of
                           the lengths of the sequences in the batch and n is the
                           number of linking functions
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, label_votes[seq_starts[i]]
                           is the row vector of labeling function outputs for the
                           first element in the ith sequence
        :return: vector of length m, where element is the most likely predicted labels
        """
        # Converts to CSR and integers to standardize input
        label_votes = sparse.csr_matrix(label_votes, dtype=np.int)
        link_votes = sparse.csr_matrix(link_votes, dtype=np.int)
        seq_starts = np.array(seq_starts, dtype=np.int)

        out = np.ndarray((label_votes.shape[0],), dtype=np.int)

        offset = 0
        for label_votes, link_votes, seq_starts in self._create_minibatches(
                label_votes, link_votes, seq_starts, 32):
            # Initializes joint log likelihood with labeling function likelihood
            jll = self._get_labeling_function_likelihoods(label_votes)
            link_cll = self._get_linking_function_likelihoods(link_votes)
            norm_start_balance = self._get_norm_start_balance()
            norm_transitions = self._get_norm_transitions()

            T = label_votes.shape[0]
            bt = torch.zeros([T, self.num_classes])
            for i in range(0, T):
                if i in seq_starts:
                    jll[i] += norm_start_balance
                else:
                    p = jll[i-1].clone().unsqueeze(1).repeat(
                        1, self.num_classes) + norm_transitions
                    p += link_cll[i]
                    jll[i] += torch.max(p, dim=0)[0]
                    bt[i, :] = torch.argmax(p, dim=0)

            seq_ends = [x - 1 for x in seq_starts] + [label_votes.shape[0] - 1]
            res = []
            j = T-1
            while j >= 0:
                if j in seq_ends:
                    res.append(torch.argmax(jll[j, :]).item())
                if j in seq_starts:
                    j -= 1
                    continue
                res.append(int(bt[j, res[-1]].item()))
                j -= 1
            res = [x + 1 for x in res]
            res.reverse()

            for i in range(len(res)):
                out[offset + i] = res[i]
            offset += len(res)
        return out

    def get_label_distribution(self, label_votes, link_votes, seq_starts):
        """Returns the unary and pairwise marginals over true labels estimated
        by the model.

        :param label_votes: m x n matrix in {0, ..., k}, where m is the sum of
                            the lengths of the sequences in the batch, n is the
                            number of labeling functions and k is the number of
                            classes
        :param link_votes: m x n matrix in {-1, 0, 1}, where m is the sum of
                           the lengths of the sequences in the batch and n is the
                           number of linking functions
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, label_votes[seq_starts[i]]
                           is the row vector of labeling function outputs for the
                           first element in the ith sequence
        :return: p_unary, p_pairwise where p_unary is a m x k matrix representing
                 the marginal distributions over individual labels, and p_pairwise
                 is a m x k x k tensor representing pairwise marginals over the
                 ith and (i+1)th labels. For the last element in a sequence, the
                 k x k matrix will be all zeros.
        """
        # Converts to CSR and integers to standardize input
        label_votes = sparse.csr_matrix(label_votes, dtype=np.int)
        link_votes = sparse.csr_matrix(link_votes, dtype=np.int)
        seq_starts = np.array(seq_starts, dtype=np.int)

        out_unary = np.zeros((label_votes.shape[0], self.num_classes))
        out_pairwise = np.zeros((label_votes.shape[0], self.num_classes, self.num_classes))

        offset = 0
        for label_votes, link_votes, seq_starts in self._create_minibatches(
                label_votes, link_votes, seq_starts, 32):
            # Computes observation likelihoods and initializes alpha and beta messages
            label_cll = self._get_labeling_function_likelihoods(label_votes)
            link_cll = self._get_linking_function_likelihoods(link_votes)
            alpha = torch.zeros(label_cll.shape)
            beta = torch.zeros(label_cll.shape)

            # Computes alpha
            next_seq = 0
            for i in range(label_votes.shape[0]):
                if next_seq == len(seq_starts) or i < seq_starts[next_seq]:
                    # i is not the start of a sequence
                    temp = alpha[i-1].unsqueeze(1).repeat(1, self.num_classes)
                    temp = temp + self._get_norm_transitions()
                    temp += link_cll[i]
                    alpha[i] = label_cll[i] + temp.logsumexp(0)
                else:
                    # i is the start of a sequence
                    alpha[i] = label_cll[i] + self._get_norm_start_balance()
                    next_seq += 1

            # Computes beta
            this_seq = seq_starts.shape[0] - 1
            beta[-1, :] = 1
            for i in range(label_votes.shape[0] - 2, -1, -1):
                if i == seq_starts[this_seq] - 1:
                    # End of sequence
                    beta[i, :] = 1
                    this_seq -= 1
                else:
                    temp = beta[i+1] + label_cll[i+1]
                    temp = temp.unsqueeze(1).repeat(1, self.num_classes)
                    temp = temp + self._get_norm_transitions()
                    temp += link_cll[i+1]
                    beta[i, :] = temp.logsumexp(0)

            # Computes p_unary
            p_unary = alpha + beta
            temp = p_unary.logsumexp(1).unsqueeze(1).repeat(1, self.num_classes)
            p_unary = p_unary - temp
            for i in range(p_unary.shape[0]):
                p = torch.exp(p_unary[i, :] - torch.max(p_unary[i, :]))
                out_unary[offset + i, :] = (p / p.sum()).detach()

            # Computes p_pairwise
            p_pairwise = torch.zeros(
                (label_votes.shape[0], self.num_classes, self.num_classes))
            for i in range(p_pairwise.shape[0] - 1):
                p_pairwise[i, :, :] = self._get_norm_transitions()
                p_pairwise[i] += alpha[i].unsqueeze(1).repeat(1, self.num_classes)
                p_pairwise[i] += label_cll[i+1].unsqueeze(0).repeat(self.num_classes, 1)
                p_pairwise[i] += beta[i+1].unsqueeze(0).repeat(self.num_classes, 1)
                p_pairwise[i] += link_cll[i+1]

                denom = p_pairwise[i].view(-1).logsumexp(0)
                denom = denom.unsqueeze(0).unsqueeze(1)
                denom = denom.repeat(self.num_classes, self.num_classes)
                p_pairwise[i] -= denom

                out_pairwise[offset + i, :, :] = torch.exp(p_pairwise[i]).detach()

            offset += label_votes.shape[0]

        return out_unary, out_pairwise

    def get_start_balance(self):
        """Returns the model's estimated class balance for the start of a
        sequence

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that the first
                 element in an example sequence has that label
        """
        return np.exp(self._get_norm_start_balance().detach().numpy())

    def get_transition_matrix(self):
        """Returns the model's estimated transition distribution from class
        label to class label in a sequence.

        :return: a k x k Numpy array, in which each element i, j is the
        probability p(c_{t+1} = j + 1 | c_{t} = i + 1)
        """
        return np.exp(self._get_norm_transitions().detach().numpy())

    def _create_minibatches(self, label_votes, link_votes, seq_starts,
                            batch_size, shuffle_seqs=False):
        if label_votes.shape[0] != link_votes.shape[0]:
            raise ValueError("label_votes and link_votes must have same number "
                             "of rows")

        # Computes explicit seq ends so that we can shuffle the sequences
        seq_ends = np.ndarray((seq_starts.shape[0],), dtype=np.int)
        for i in range(1, seq_starts.shape[0]):
            seq_ends[i - 1] = seq_starts[i] - 1
        seq_ends[-1] = label_votes.shape[0] - 1

        # Shuffles the sequences by shuffling the start and end index vectors
        if shuffle_seqs:
            index = np.arange(np.shape(seq_starts)[0])
            np.random.shuffle(index)
            seq_starts = seq_starts[index]
            seq_ends = seq_ends[index]

        # Splits seq_starts
        seq_start_batches = [np.array(
            seq_starts[i * batch_size: ((i + 1) * batch_size)],
            copy=True)
            for i in range(int(np.ceil(len(seq_starts) / batch_size)))
        ]
        seq_start_batches[-1] = np.concatenate((seq_start_batches[-1],
                                                [label_votes.shape[0]]))

        # Splits seq_ends
        seq_end_batches = [
            np.array(seq_ends[i * batch_size: ((i + 1) * batch_size + 1)], copy=True)
            for i in range(int(np.ceil(len(seq_ends) / batch_size)))
        ]
        seq_end_batches[-1] = np.concatenate((seq_end_batches[-1],
                                              [label_votes.shape[0]]))

        # Builds label_vote_batches, link_vote_batches and relative seq_start_batches
        label_vote_batches = []
        link_vote_batches = []
        rel_seq_start_batches = []
        for seq_start_batch, seq_end_batch in zip(seq_start_batches, seq_end_batches):
            label_vote_batch = []
            link_vote_batch = []
            rel_seq_start_batch = np.zeros((len(seq_start_batch),), dtype=np.int)
            total_len = 0
            for i, (start, end) in enumerate(zip(seq_start_batch, seq_end_batch)):
                label_vote_batch.append(label_votes[start:end + 1])
                link_vote_batch.append(link_votes[start:end + 1])
                rel_seq_start_batch[i] = total_len
                total_len += end - start + 1
            label_vote_batches.append(
                sparse.coo_matrix(sparse.vstack(label_vote_batch), copy=True))
            link_vote_batches.append(
                sparse.coo_matrix(sparse.vstack(link_vote_batch), copy=True))
            rel_seq_start_batches.append(rel_seq_start_batch)

        return list(zip(label_vote_batches, link_vote_batches, rel_seq_start_batches))

    def _get_linking_function_likelihoods(self, votes):
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        cll = torch.zeros((votes.shape[0], self.num_classes, self.num_classes))

        # Initializes normalizing constants
        z_prop = self.link_propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_linking_funcs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        z_acc = self.link_accuracy.unsqueeze(1)
        z_acc = torch.cat((z_acc, -1 * self.link_accuracy.unsqueeze(1)), dim=1)
        z_acc = torch.logsumexp(z_acc, dim=1)

        # Subtracts normalizing constant for propensities from cll
        # (since it applies to all outcomes)
        cll -= torch.sum(z_prop)

        # Loops over votes and classes to compute conditional log-likelihood
        for i, j, v in zip(votes.row, votes.col, votes.data):
            if v != 1 and v != -1:
                continue

            for k1 in range(self.num_classes):
                for k2 in range(self.num_classes):
                    if k1 == k2:
                        if v == 1:
                            cll[i, k1, k2] += self.link_propensity[j]
                            cll[i, k1, k2] += self.link_accuracy[j]
                            cll[i, k1, k2] -= z_acc[j]
                        else:
                            cll[i, k1, k2] += self.link_propensity[j]
                            cll[i, k1, k2] -= self.link_accuracy[j]
                            cll[i, k1, k2] -= z_acc[j]
                    else:
                        if v == 1:
                            cll[i, k1, k2] += self.link_propensity[j]
                            cll[i, k1, k2] -= self.link_accuracy[j]
                            cll[i, k1, k2] -= z_acc[j]
                        else:
                            cll[i, k1, k2] += self.link_propensity[j]
                            cll[i, k1, k2] += self.link_accuracy[j]
                            cll[i, k1, k2] -= z_acc[j]

        return cll

    def _get_regularization_loss(self):
        neg_entropy = 0.0

        # Start balance
        norm_start_balance = self._get_norm_start_balance()
        exp_class_balance = torch.exp(norm_start_balance)
        for k in range(self.num_classes):
            neg_entropy += norm_start_balance[k] * exp_class_balance[k]

        # Transitions
        norm_transitions = self._get_norm_transitions()
        for i in range(self.num_classes):
            exp_transitions = torch.exp(norm_transitions[i])
            for k in range(self.num_classes):
                neg_entropy += norm_transitions[i, k] * exp_transitions[k]

        entropy_prior = self.balance_prior * neg_entropy

        # Accuracy prior
        acc = torch.cat((self.accuracy.view(-1), self.link_accuracy))
        acc_prior = self.acc_prior * torch.norm(acc - self.init_acc)

        return acc_prior + entropy_prior

    def _get_norm_start_balance(self):
        return self.start_balance - self.start_balance.logsumexp(0)

    def _get_norm_transitions(self):
        denom = self.transitions.logsumexp(1).unsqueeze(1).repeat(1, self.num_classes)
        return self.transitions - denom
