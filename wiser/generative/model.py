from wiser.generative import get_label_to_ix, get_rules, train_generative_model, evaluate_generative_model, clean_inputs
from wiser.data import save_label_distribution
from wiser.eval import get_generative_model_inputs


class Model:

    def __init__(self, model_module, train_data, dev_data, test_data, init_acc=0.9, acc_prior=100, balance_prior=500):
        """
        Initializes a predefined generative model using the given parameters

        :param model:           labelmodel class of generative model to be initialized
        :param train_data:      array of  AllenNLP instances used as training samples
        :param dev_data:        array of labeled AllenNLP instances used as development samples
        :param test_data:       array of labeled AllenNLP instances used as testing samples
        :param init_acc:        initial estimated tagging and linking rule accuracy,
                                also used as the mean of the prior distribution of the model parameters
        :param acc_prior:       weight of the regularizer  pulling  tagging  and  linking  rule  accuracies
                                toward their initial values.
        :param balance_prior:   used to regularize the class prior in naiveBayes or the initial class distribution
                                for HMM and linkedHMM, as well as the transition matrix in those methods,
                                towards a more uniform distribution.
        """

        self.model_type = model_module.__name__
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.gen_label_to_ix, self.disc_label_to_ix = get_label_to_ix(train_data+dev_data)
        tagging_rules, linking_rules = get_rules(train_data+dev_data)

        if self.model_type == 'NaiveBayes':
            self.model = model_module(len(self.gen_label_to_ix) - 1, len(tagging_rules),
                               init_acc, acc_prior, balance_prior)
        elif self.model_type == 'HMM' or self.model_type == 'LinkedHMM':
            self.model = model_module(len(self.gen_label_to_ix) - 1, len(tagging_rules), len(linking_rules),
                               init_acc, acc_prior, balance_prior)
        else:
            raise ValueError("Unknown model type: %s" % str(type(self.model)))

    def train(self, config):
        """
        Trains the generative model

        :param config:          labelmodel config specifying training configuration
        """
        p, r, f1 = train_generative_model(self.model, self.train_data, self.dev_data,
                                          label_to_ix=self.gen_label_to_ix, config=config)

        # Prints train precision, recall, and F1 scores
        return p, r, f1

    def evaluate(self, on_test=True):
        """
        Evaluates the generative model

        :param on_test: boolean indicatin gwhether to evaluate the generative model on test
                        (if false, the function will evaluate on dev)
        """
        if on_test:
            data = self.test_data
        else:
            data = self.dev_data

        return evaluate_generative_model(model=self.model, data=data, label_to_ix=self.gen_label_to_ix)

    def save_probabilistic_output(self, path, save_tags=True):

        """
        Saves the probabilistic output of the generative model

        :param path: path to save the training, development, and testint data
        :param save_tags: boolean indicating whether to save the true tags, if any
        """

        inputs = clean_inputs(get_generative_model_inputs(self.train_data, self.gen_label_to_ix), self.model)

        if self.model_type == "NaiveBayes":
            dist = self.model.get_label_distribution(*inputs)

            save_label_distribution(path + '/train_data.p',
                                    self.train_data,
                                    unary_marginals=dist,
                                    gen_label_to_ix=self.gen_label_to_ix,
                                    disc_label_to_ix=self.disc_label_to_ix,
                                    save_tags=save_tags)

        else:
            p_unary, p_pairwise = self.model.get_label_distribution(*inputs)

            save_label_distribution(path + '/train_data.p',
                                    self.train_data,
                                    unary_marginals=p_unary,
                                    pairwise_marginals=p_pairwise,
                                    gen_label_to_ix=self.gen_label_to_ix,
                                    disc_label_to_ix=self.disc_label_to_ix,
                                    save_tags=save_tags)

        save_label_distribution(path + '/dev_data.p', self.dev_data)
        save_label_distribution(path + '/test_data.p', self.test_data)





