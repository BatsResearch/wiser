import copy
import numpy as np
from wiser.eval import score_predictions
from collections import Counter


def grid_search(model_constructor, train_data, dev_data, config,
                acc_priors, balance_priors, epochs, label_to_ix):
    best_p = float('-inf')
    best_r = float('-inf')
    best_f1 = float('-inf')
    best_params = None
    best_acc_prior = None
    best_balance_prior = None

    for acc_prior in acc_priors:
        for balance_prior in balance_priors:
            model = model_constructor(acc_prior, balance_prior)
            p, r, f1 = train_generative_model(
                model, train_data, dev_data, epochs, label_to_ix, config)
            if f1 > best_f1:
                best_p = p
                best_r = r
                best_f1 = f1
                best_params = model.state_dict()
                best_acc_prior = acc_prior
                best_balance_prior = balance_prior

    best_model = model_constructor(best_acc_prior, best_balance_prior)
    best_model.load_state_dict(best_params)

    return best_model, best_p, best_r, best_f1


def get_label_to_ix(data):
    tag_count = Counter()

    for instance in data:
        for tag in instance['tags']:
            tag_count[tag] += 1

    disc_label_to_ix = {value[0]: int(ix) for ix, value in enumerate(tag_count.most_common())}
    gen_label_to_ix = disc_label_to_ix.copy()

    for ix in gen_label_to_ix:
        gen_label_to_ix[ix] += 1
    gen_label_to_ix['ABS'] = 0

    return gen_label_to_ix, disc_label_to_ix


def get_rules(data):
    labeling_functions = set()
    linking_functions = set()
    for doc in data:
        for name in doc['WISER_LABELS'].keys():
            labeling_functions.add(name)
        for name in doc['WISER_LINKS'].keys():
            linking_functions.add(name)

    return labeling_functions, linking_functions


def train_generative_model(model, train_data, dev_data, label_to_ix, config):
    train_inputs = clean_inputs(get_generative_model_inputs(train_data, label_to_ix), model)

    best_p = float('-inf')
    best_r = float('-inf')
    best_f1 = float('-inf')
    best_params = None

    for i in range(config.epochs):
        model.estimate_label_model(*train_inputs, config=config)
        results = evaluate_generative_model(model, dev_data, label_to_ix)

        if results["F1"][0] > best_f1:
            best_p = results["P"][0]
            best_r = results["R"][0]
            best_f1 = results["F1"][0]
            best_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_params)

    return best_p, best_r, best_f1


def evaluate_generative_model(model, data, label_to_ix):

    inputs = clean_inputs(get_generative_model_inputs(data, label_to_ix), model)
    ix_to_label = dict(map(reversed, label_to_ix.items()))
    predictions = model.get_most_probable_labels(*inputs)
    label_predictions = [ix_to_label[ix] for ix in predictions]
    return score_predictions(data, label_predictions)


def clean_inputs(inputs, model):
    if type(model).__name__ == "NaiveBayes":
        inputs = (inputs[0],)
    elif type(model).__name__ == "HMM":
        inputs = (inputs[0], inputs[2])
    elif type(model).__name__ == "LinkedHMM":
        pass
    else:
        raise ValueError("Unknown model type: %s" % str(type(model)))
    return inputs


def get_unweighted_training_labels(instance, label_to_ix, treat_tie_as):
    maj_vote = [None] * len(instance['tokens'])

    for i in range(len(instance['tokens'])):
        # Collects the votes for the ith token
        votes = {}
        for lf_labels in instance['WISER_LABELS'].values():
            if lf_labels[i] not in votes:
                votes[lf_labels[i]] = 0
            votes[lf_labels[i]] += 1

        # Takes the majority vote, not counting abstentions
        try:
            del votes['ABS']
        except KeyError:
            pass

        if len(votes) == 0:
            maj_vote[i] = treat_tie_as
        elif len(votes) == 1:
            maj_vote[i] = list(votes.keys())[0]
        else:
            sort = sorted(votes.keys(), key=lambda x: votes[x], reverse=True)
            first, second = sort[0:2]
            if votes[first] == votes[second]:
                maj_vote[i] = treat_tie_as
            else:
                maj_vote[i] = first


def get_generative_model_inputs(instances, label_to_ix):
    label_name_to_col = {}
    link_name_to_col = {}

    # Collects label and link function names
    names = set()
    for doc in instances:
        if 'WISER_LABELS' in doc:
            for name in doc['WISER_LABELS']:
                names.add(name)
    for name in sorted(names):
        label_name_to_col[name] = len(label_name_to_col)

    names = set()
    for doc in instances:
        if 'WISER_LINKS' in doc:
            for name in doc['WISER_LINKS']:
                names.add(name)
    for name in sorted(names):
        link_name_to_col[name] = len(link_name_to_col)

    # Counts total tokens
    total_tokens = 0
    for doc in instances:
        total_tokens += len(doc['tokens'])

    # Initializes output data structures
    label_votes = np.zeros((total_tokens, len(label_name_to_col)), dtype=np.int)
    link_votes = np.zeros((total_tokens, len(link_name_to_col)), dtype=np.int)
    seq_starts = np.zeros((len(instances),), dtype=np.int)

    # Populates outputs
    offset = 0
    for i, doc in enumerate(instances):
        seq_starts[i] = offset

        for name in sorted(doc['WISER_LABELS'].keys()):
            for j, vote in enumerate(doc['WISER_LABELS'][name]):
                label_votes[offset + j, label_name_to_col[name]] = label_to_ix[vote]

        for name in sorted(doc['WISER_LINKS'].keys()):
            for j, vote in enumerate(doc['WISER_LINKS'][name]):
                link_votes[offset + j, link_name_to_col[name]] = vote

        offset += len(doc['tokens'])

    return label_votes, link_votes, seq_starts
