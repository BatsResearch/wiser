from wiser.generative import get_label_to_ix, get_rules, train_generative_model, evaluate_generative_model, clean_inputs, get_predictions_generative_model
from wiser.data import save_label_distribution
from wiser.eval import get_generative_model_inputs


class Model:

    def __init__(self, model_module, init_acc=0.9, acc_prior=50, balance_prior=100):
        """
        Initializes a predefined generative model using the given parameters

        :param model_module         labelmodel class of generative model to be initialized
        :param init_acc:            initial estimated tagging and linking rule accuracy,
                                    also used as the mean of the prior distribution of the model parameters
        :param acc_prior:           weight of the regularizer  pulling  tagging  and  linking  rule  accuracies
                                    toward their initial values.
        :param balance_prior:       used to regularize the class prior in naiveBayes or the initial class distribution
                                    for HMM and linkedHMM, as well as the transition matrix in those methods,
                                    towards a more uniform distribution.
        """

        self.model_module = model_module
        self.model_type = model_module.__name__
        self.init_acc = init_acc
        self.acc_prior = acc_prior
        self.balance_prior = balance_prior

        self.gen_label_to_ix = None
        self.disc_label_to_ix = None
        self.model = None

    def train(self, config, train_data, dev_data):
        """
        Trains the generative model

        :param train_data:          array of  AllenNLP instances used as training samples
        :param dev_data:            array of labeled AllenNLP instances used as development samples
        :param config:              labelmodel config specifying training configuration
        """

        self.gen_label_to_ix, self.disc_label_to_ix = get_label_to_ix(train_data + dev_data)
        tagging_rules, linking_rules = get_rules(train_data + dev_data)

        if self.model_type == 'NaiveBayes' or self.model_type == 'HMM':
            self.model = self.model_module(len(self.gen_label_to_ix) - 1, len(tagging_rules),
                                      self.init_acc, self.acc_prior, self.balance_prior)
        elif self.model_type == 'LinkedHMM':
            self.model = self.model_module(len(self.gen_label_to_ix) - 1, len(tagging_rules), len(linking_rules),
                                      self.init_acc, self.acc_prior, self.balance_prior)
        else:
            raise ValueError("Unknown model type: %s" % str(type(self.model_type)))

        p, r, f1 = train_generative_model(self.model, train_data, dev_data,
                                          label_to_ix=self.gen_label_to_ix, config=config)

        # Prints development precision, recall, and F1 scores
        return p, r, f1

    def evaluate(self, data):
        """
        Evaluates the generative model

        :param data:                array of labeled AllenNLP instances used as evaluation samples

        """

        if self.model is None:
            raise ValueError("You need to train the generative model before evaluating it's output.")

        return evaluate_generative_model(model=self.model, data=data, label_to_ix=self.gen_label_to_ix)

    
    def get_predictions(self, data):
        """
        Gets predictions for the generative model for an input sentence

        :param data:                array of labeled AllenNLP instances used as evaluation samples

        """

        if self.model is None:
            raise ValueError("You need to train the generative model before evaluating it's output.")

        return get_predictions_generative_model(model=self.model, data=data, label_to_ix=self.gen_label_to_ix)

    def save_output(self, data, path, save_distribution=True, save_tags=True):

        """
        Saves the probabilistic output of the generative model

        :param data:                array of labeled AllenNLP instances used as evaluation samples
        :param path:                path to save the data to
        :param save_distribution:   boolean indicating to save the probabilistic distrubution
        :param save_tags:           boolean indicating to save the true tags (if any)
        """

        if self.model is None:
            raise ValueError("You need to train the generative model before saving the output to disk.")

        inputs = clean_inputs(get_generative_model_inputs(data, self.gen_label_to_ix), self.model)

        if save_distribution:
            if self.model_type == "NaiveBayes":
                dist = self.model.get_label_distribution(*inputs)

                save_label_distribution(path,
                                        data,
                                        unary_marginals=dist,
                                        gen_label_to_ix=self.gen_label_to_ix,
                                        disc_label_to_ix=self.disc_label_to_ix,
                                        save_tags=save_tags)

            else:
                p_unary, p_pairwise = self.model.get_label_distribution(*inputs)

                save_label_distribution(path,
                                        data,
                                        unary_marginals=p_unary,
                                        pairwise_marginals=p_pairwise,
                                        gen_label_to_ix=self.gen_label_to_ix,
                                        disc_label_to_ix=self.disc_label_to_ix,
                                        save_tags=save_tags)
        else:
            save_label_distribution(path, data, save_tags)





