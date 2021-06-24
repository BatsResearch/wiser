import numpy as np
import pandas as pd
from wiser.viewer import Viewer
from allennlp.data import Instance

def score_labels_majority_vote(instances,  gold_label_key='tags',
                               treat_tie_as='O', span_level=True):
    tp, fp, fn = 0, 0, 0
    for instance in instances:
        maj_vote = _get_label_majority_vote(instance, treat_tie_as)
        if span_level:
            score = _score_sequence_span_level(maj_vote, instance[gold_label_key])
        else:
            score = _score_sequence_token_level(maj_vote, instance[gold_label_key])
        tp += score[0]
        fp += score[1]
        fn += score[2]

    # Collects results into a dataframe
    column_names = ["TP", "FP", "FN", "P", "R", "F1"]
    p, r, f1 = _get_p_r_f1(tp, fp, fn)
    record = [tp, fp, fn, p, r, f1]
    index = ["Majority Vote"] if span_level else ["Majority Vote (Token Level)"]
    results = pd.DataFrame.from_records(
        [record], columns=column_names, index=index)
    results = pd.DataFrame.sort_index(results)
    return results


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
        
        if 'WISER_LINKS' in doc:
            for name in sorted(doc['WISER_LINKS'].keys()):
                for j, vote in enumerate(doc['WISER_LINKS'][name]):
                    link_votes[offset + j, link_name_to_col[name]] = vote

        offset += len(doc['tokens'])

    return label_votes, link_votes, seq_starts


def score_predictions(instances, predictions,
                      gold_label_key='tags', span_level=True):

    tp, fp, fn = 0, 0, 0

    offset = 0
    for instance in instances:
        length = len(instance[gold_label_key])
        if span_level:
            scores = _score_sequence_span_level(
                predictions[offset:offset+length], instance[gold_label_key])
        else:
            scores = _score_sequence_token_level(
                predictions[offset:offset+length], instance[gold_label_key])
        tp += scores[0]
        fp += scores[1]
        fn += scores[2]
        offset += length

    # Collects results into a dataframe
    column_names = ["TP", "FP", "FN", "P", "R", "F1"]
    p = round(tp / (tp + fp) if tp > 0 or fp > 0 else 0.0, ndigits=4)
    r = round(tp / (tp + fn) if tp > 0 or fn > 0 else 0.0, ndigits=4)
    f1 = round(2 * p * r / (p + r) if p > 0 and r > 0 else 0.0, ndigits=4)
    record = [tp, fp, fn, p, r, f1]
    index = ["Predictions"] if span_level else ["Predictions (Token Level)"]
    results = pd.DataFrame.from_records(
        [record], columns=column_names, index=index)
    results = pd.DataFrame.sort_index(results)
    return results


def score_tagging_rules(instances, gold_label_key='tags'):
    lf_scores = {}
    for instance in instances:
        for lf_name, predictions in instance['WISER_LABELS'].items():
            if lf_name not in lf_scores:
                # Initializes true positive, false positive, false negative,
                # correct, and total vote counts
                lf_scores[lf_name] = [0, 0, 0, 0, 0]

            scores = _score_sequence_span_level(predictions, instance[gold_label_key])
            lf_scores[lf_name][0] += scores[0]
            lf_scores[lf_name][1] += scores[1]
            lf_scores[lf_name][2] += scores[2]

            scores = _score_token_accuracy(predictions, instance[gold_label_key])
            lf_scores[lf_name][3] += scores[0]
            lf_scores[lf_name][4] += scores[1]

    # Computes accuracies
    for lf_name in lf_scores.keys():
        if lf_scores[lf_name][3] > 0:
            lf_scores[lf_name][3] = float(lf_scores[lf_name][3]) / lf_scores[lf_name][4]
            lf_scores[lf_name][3] = round(lf_scores[lf_name][3], ndigits=4)
        else:
            lf_scores[lf_name][3] = float('NaN')

    # Collects results into a dataframe
    column_names = ["TP", "FP", "FN", "Token Acc.", "Token Votes"]
    results = pd.DataFrame.from_dict(lf_scores, orient="index", columns=column_names)
    results = pd.DataFrame.sort_index(results)
    return results


def score_linking_rules(instances, gold_label_keys='tags'):
    lf_scores = {}
    for instance in instances:
        for lf_name, predictions in instance['WISER_LINKS'].items():
            if lf_name not in lf_scores:
                # Initializes counts for correct entity links, correct
                # non-entity links, and incorrect links
                lf_scores[lf_name] = [0, 0, 0]

            for i in range(1, len(predictions)):
                if predictions[i] == 1:
                    entity0 = instance[gold_label_keys][i-1][0] == 'I'
                    entity0 = entity0 or instance[gold_label_keys][i-1][0] == 'B'

                    entity1 = instance[gold_label_keys][i][0] == 'I'
                    entity1 = entity1 or instance[gold_label_keys][i][0] == 'B'

                    if entity0 and entity1:
                        lf_scores[lf_name][0] += 1
                    elif not entity0 and not entity1:
                        lf_scores[lf_name][1] += 1
                    else:
                        lf_scores[lf_name][2] += 1

    for counts in lf_scores.values():
        if counts[0] + counts[1] + counts[2] == 0:
            counts.append(float('NaN'))
        else:
            counts.append(round(
                (counts[0] + counts[1]) / (counts[0] + counts[1] + counts[2]), ndigits=4))

    # Collects results into a dataframe
    column_names = ["Entity Links", "Non-Entity Links", "Incorrect Links", "Accuracy"]
    results = pd.DataFrame.from_dict(lf_scores, orient="index", columns=column_names)
    results = pd.DataFrame.sort_index(results)
    return results


def get_mv_label_distribution(instances, label_to_ix, treat_tie_as):
    distribution = []
    for instance in instances:
        mv = _get_label_majority_vote(instance, treat_tie_as)
        for vote in mv:
            p = [0.0] * len(label_to_ix)
            p[label_to_ix[vote]] = 1.0
            distribution.append(p)
    return np.array(distribution)


def get_unweighted_label_distribution(instances, label_to_ix, treat_abs_as):
    # Counts votes
    distribution = []
    for instance in instances:
        for i in range(len(instance['tokens'])):
            votes = [0] * len(label_to_ix)
            for vote in instance['WISER_LABELS'].values():
                if vote[i] != "ABS":
                    votes[label_to_ix[vote[i]]] += 1
            distribution.append(votes)

    # For each token, adds one vote for the default if there are none
    distribution = np.array(distribution)
    for i, check in enumerate(distribution.sum(axis=1) == 0):
        if check:
            distribution[i, label_to_ix[treat_abs_as]] = 1

    # Normalizes the counts
    distribution = distribution / np.expand_dims(distribution.sum(axis=1), 1)

    return distribution


def _score_sequence_span_level(predicted_labels, gold_labels):
    if len(predicted_labels) != len(gold_labels):
        raise ValueError("Lengths of predicted_labels and gold_labels must match")

    tp, fp, fn = 0, 0, 0
    # Collects predicted and correct spans for the instance
    predicted_spans, correct_spans = set(), set()
    data = ((predicted_labels, predicted_spans), (gold_labels, correct_spans))
    for labels, spans in data:
        start = None
        tag = None
        for i in range(len(labels)):
            if labels[i][0] == 'I':
                # Two separate conditional statements so that 'I' is always
                # recognized as a valid label
                if start is None:
                    start = i
                    tag = labels[i]
                # Also checks if label has switched to new type
                elif tag != labels[i]:
                    spans.add((start, i, tag))
                    start = i
                    tag = labels[i]
            elif labels[i][0] == 'O' or labels[i] == 'ABS':
                if start is not None:
                    spans.add((start, i, tag))
                start = None
                tag = None
            elif labels[i][0] == 'B':
                if start is not None:
                    spans.add((start, i, tag))
                start = i
                tag = labels[i]
            else:
                raise ValueError("Unrecognized label: %s" % labels[i] )

        # Closes span if still active
        if start is not None:
            spans.add((start, len(labels), tag))

    # Compares predicted spans with correct spans
    for span in correct_spans:
        if span in predicted_spans:
            tp += 1
            predicted_spans.remove(span)
        else:
            fn += 1
    fp += len(predicted_spans)

    return tp, fp, fn


def _score_sequence_token_level(predicted_labels, gold_labels):
    if len(predicted_labels) != len(gold_labels):
        raise ValueError("Lengths of predicted_labels and gold_labels must match")

    tp, fp, fn = 0, 0, 0
    for i in range(len(predicted_labels)):
        prediction = predicted_labels[i]
        gold = gold_labels[i]

        if gold[0] == 'I' or gold[0] == 'B':
            if prediction == gold:
                tp += 1
            elif prediction[0] == 'I' or prediction[0] == 'B':
                fp += 1
                fn += 1
            else:
                fn += 1
        elif prediction[0] == 'I' or prediction[0] == 'B':
            fp += 1

    return tp, fp, fn


def _score_token_accuracy(predicted_labels, gold_labels):
    if len(predicted_labels) != len(gold_labels):
        raise ValueError("Lengths of predicted_labels and gold_labels must match")

    correct = 0
    votes = 0

    for i in range(len(gold_labels)):
        if predicted_labels[i] == gold_labels[i]:
            correct += 1
        if predicted_labels[i] != 'ABS':
            votes += 1

    return correct, votes


def _get_label_majority_vote(instance, treat_tie_as):
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
    return maj_vote


def _get_p_r_f1(tp, fp, fn):
    p = round(tp / (tp + fp) if tp > 0 or fp > 0 else 0.0, ndigits=4)
    r = round(tp / (tp + fn) if tp > 0 or fn > 0 else 0.0, ndigits=4)
    f1 = round(2 * p * r / (p + r) if p > 0 or r > 0 else 0.0, ndigits=4)

    return p, r, f1

    
def tagging_rule_errors(instances, rule, error_type='fn', gold_label_key='tags', mode = 'span'):
    if not error_type in {'fn', 'fp', 'both'}:
        raise IllegalArgumentException('Error_type must be one of \'fn\', \'fp\' or \'both\'')
    if not mode in {'span', 'token'}:
        raise IllegalArgumentException('Mode must be one of \'span\' or \'token\'')

    data = []
    for instance in instances:
        predictions = instance['WISER_LABELS'][rule]
        if mode == 'span':
            scores = _score_sequence_span_level(predictions, instance[gold_label_key])
        elif mode == 'token':
            scores = _score_sequence_token_level(predictions, instance[gold_label_key])

        if(scores[1] > 0 and error_type=='fp'):
            data.append(instance)

        elif(scores[2] > 0 and error_type == 'fn'):
            data.append(instance)

        elif (scores[1] > 0 or scores[2] > 0) and error_type == 'both':
            data.append(instance)


    # Collects results into an Instance
    data = Instance(data)
    return data