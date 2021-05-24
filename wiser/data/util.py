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
            pairwise_marginals_list.append(pairwise_marginals[i])
        i += 1

    return [unary_marginals_list, pairwise_marginals_list, i]


def get_complete_unary_marginals(unary_marginals, gen_label_to_ix, disc_label_to_ix):

    if unary_marginals is None or gen_label_to_ix is None or disc_label_to_ix is None:
        return unary_marginals

    new_unaries = np.zeros((len(unary_marginals), len(disc_label_to_ix)))

    for k, v in disc_label_to_ix.items():
        if k in gen_label_to_ix:
            new_unaries[:, v] = unary_marginals[:, gen_label_to_ix[k]-1]

    return new_unaries


def get_complete_pairwise_marginals(pairwise_marginals, gen_label_to_ix, disc_label_to_ix):

    if pairwise_marginals is None or gen_label_to_ix is None or disc_label_to_ix is None:
        return pairwise_marginals

    new_pairwise = np.zeros((len(pairwise_marginals), len(disc_label_to_ix), len(disc_label_to_ix)))

    for k1, v1 in disc_label_to_ix.items():
        for k2, v2 in disc_label_to_ix.items():
            if k1 in gen_label_to_ix and k2 in gen_label_to_ix:
                new_pairwise[:, v1, v2] = pairwise_marginals[:, gen_label_to_ix[k1]-1, gen_label_to_ix[k2]-1]

    return new_pairwise


def save_label_distribution(save_path, data, unary_marginals=None,
                            pairwise_marginals=None, gen_label_to_ix=None,
                            disc_label_to_ix=None, save_tags=True):

    unary_marginals = get_complete_unary_marginals(unary_marginals,
                                                   gen_label_to_ix,
                                                   disc_label_to_ix)

    pairwise_marginals = get_complete_pairwise_marginals(pairwise_marginals,
                                                      gen_label_to_ix,
                                                      disc_label_to_ix)

    i = 0
    instances = []
    for instance in data:
        instance_tokens = instance['tokens']
        fields = {'tokens': instance_tokens}

        if 'sentence_spans' in instance:
            fields['sentence_spans'] = instance['sentence_spans']

        if 'tags' in instance and save_tags:
            fields['tags'] = instance['tags']

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
