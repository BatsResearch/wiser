from allennlp.data.fields import ArrayField
from allennlp.data import Instance
import numpy as np
import pickle


def get_vote_mask(instance):
    dict_items = instance['WISER_LABELS'].items()
    votes = np.array([item[1] for item in dict_items])
    mask = np.where(votes == 'ABS', 0, 1)
    return ArrayField(np.ndarray.max(mask, 0))

def get_marginals(i, num_tokens, unary_marginals, pairwise_marginals):
    unary_marginals_list = []
    pairwise_marginals_list = None if pairwise_marginals is None else []

    for _ in range(num_tokens):
        unary_marginals_list.append(unary_marginals[i])

        if pairwise_marginals is not None:
            pairwise_marginals_list.append(pairwise_marginals)
        i += 1

    if pairwise_marginals is not None:
        del pairwise_marginals_list[-1]

    return [unary_marginals_list, pairwise_marginals_list, i]

def save_label_distribution(save_path, data, unary_marginals=None, pairwise_marginals=None, save_tags=True):
    
    i = 0
    instances = []
    for instance in data:
        instance_tokens = instance['tokens']
        fields = {'tokens': instance_tokens}

        if 'sentence_spans' in instance:
            fields['sentence_spans'] = instance['sentence_spans']

        if 'tags' in instance and save_tags:
            fields['tags'] =  instance['tags']

        if unary_marginals is not None:
            instance_unary_list, instance_pairwise_list, i = get_marginals(
                i, len(instance_tokens), unary_marginals, pairwise_marginals)

            fields['unary_marginals'] = ArrayField(np.array(instance_unary_list))

            if instance_pairwise_list is not None:
                fields['pairwise_marginals'] = ArrayField(np.array(instance_pairwise_list))

        fields['vote_mask'] = get_vote_mask(instance)

        instances.append(Instance(fields))

    with open(save_path, 'wb') as f:
        pickle.dump(instances, f)
