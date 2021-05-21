from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from typing import Iterator, Dict, List
from allennlp.data.fields import ArrayField, TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
import pickle
import os

@DatasetReader.register('weak_label')
class WeakLabelDatasetReader(DatasetReader):
    """
    DatasetReader for CDR corpus available at
    https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus
    This is an abstract class. Concrete subclasses need to implement _read by
    calling _cdr_read with a second argument that is the name of the entity type
    to include in the data set.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, split_sentences: bool = False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.split_sentences = split_sentences

    def text_to_instance(self, tokens: List[Token]) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        if(os.path.isdir(file_path)):

            for filename in os.listdir(file_path):
                datapath = os.path.join(file_path, filename)

                with open(datapath, 'rb') as f:
                    data = pickle.load(f)

                if not self.split_sentences:
                    for instance in data:
                        tokens = [token for token in instance['tokens']]
                        tokens_field = TextField(tokens, self.token_indexers)
                        instance.add_field('tokens', tokens_field)
                        yield instance
                else:
                    for instance in data:
                        if 'sentence_spans' not in instance:
                            raise ValueError("No sentence spans detected in the dataset "
                                            "you're attempting to read. "
                                            "Did you forget to generate them?")

                        tokens_field = instance['tokens']
                        tags_field = instance['tags'] if 'tags' in instance else None
                        unary_marginals_field = instance['unary_marginals'] if 'unary_marginals' in instance else None
                        pairwise_marginals_field = instance['pairwise_marginals'] if 'pairwise_marginals' in instance else None
                        vote_mask_field = instance['vote_mask'] if 'vote_mask' in instance else None

                        tokens = [token for token in tokens_field]
                        tags = [tag for tag in tags_field]
                        unary_marginals, pairwise_marginals, vote_mask = [None, None, None]

                        if unary_marginals_field:
                            unary_marginals = unary_marginals_field.as_tensor(unary_marginals_field.get_padding_lengths()).numpy()
                        if pairwise_marginals_field:
                            pairwise_marginals = pairwise_marginals_field.as_tensor(pairwise_marginals_field.get_padding_lengths()).numpy()
                        if vote_mask_field:
                            vote_mask = vote_mask_field.as_tensor(vote_mask_field.get_padding_lengths()).numpy()

                        sentence_delimiters = instance['sentence_spans'].metadata

                        for delimiter in sentence_delimiters:
                            sentence_tokens = tokens[delimiter[0]:delimiter[1]]

                            if len(sentence_tokens) == 0:
                                continue

                            sentence_tokens_field = TextField(sentence_tokens, self.token_indexers)
                            fields = {"tokens": sentence_tokens_field}

                            if tags is not None:
                                sentence_tags =tags[delimiter[0]:delimiter[1]]
                                assert len(sentence_tags) == len(sentence_tokens)
                                fields["tags"] = SequenceLabelField(labels=sentence_tags, sequence_field=sentence_tokens_field)

                            if unary_marginals is not None:
                                sentence_unary_marginals = unary_marginals[delimiter[0]:delimiter[1]]
                                assert len(sentence_unary_marginals) == len(sentence_tokens)
                                fields['unary_marginals'] = ArrayField(sentence_unary_marginals)

                            if pairwise_marginals is not None:
                                sentence_pairwise_marginals = pairwise_marginals[delimiter[0]:delimiter[1]]
                                assert len(sentence_pairwise_marginals) == len(sentence_tokens)
                                fields['pairwise_marginals'] = ArrayField(sentence_pairwise_marginals)

                            if vote_mask is not None:
                                sentence_vote_mask = vote_mask[delimiter[0]:delimiter[1]]
                                assert len(sentence_vote_mask) == len(sentence_tokens)
                                fields["vote_mask"] = ArrayField(sentence_vote_mask)

                            yield Instance(fields)

        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if not self.split_sentences:
                for instance in data:
                    tokens = [token for token in instance['tokens']]
                    tokens_field = TextField(tokens, self.token_indexers)
                    instance.add_field('tokens', tokens_field)
                    yield instance
            else:
                for instance in data:
                    if 'sentence_spans' not in instance:
                        raise ValueError("No sentence spans detected in the dataset "
                                        "you're attempting to read. "
                                        "Did you forget to generate them?")

                    tokens_field = instance['tokens']
                    tags_field = instance['tags'] if 'tags' in instance else None
                    unary_marginals_field = instance['unary_marginals'] if 'unary_marginals' in instance else None
                    pairwise_marginals_field = instance['pairwise_marginals'] if 'pairwise_marginals' in instance else None
                    vote_mask_field = instance['vote_mask'] if 'vote_mask' in instance else None

                    tokens = [token for token in tokens_field]
                    tags = [tag for tag in tags_field]
                    unary_marginals, pairwise_marginals, vote_mask = [None, None, None]

                    if unary_marginals_field:
                        unary_marginals = unary_marginals_field.as_tensor(unary_marginals_field.get_padding_lengths()).numpy()
                    if pairwise_marginals_field:
                        pairwise_marginals = pairwise_marginals_field.as_tensor(pairwise_marginals_field.get_padding_lengths()).numpy()
                    if vote_mask_field:
                        vote_mask = vote_mask_field.as_tensor(vote_mask_field.get_padding_lengths()).numpy()

                    sentence_delimiters = instance['sentence_spans'].metadata

                    for delimiter in sentence_delimiters:
                        sentence_tokens = tokens[delimiter[0]:delimiter[1]]

                        if len(sentence_tokens) == 0:
                            continue

                        sentence_tokens_field = TextField(sentence_tokens, self.token_indexers)
                        fields = {"tokens": sentence_tokens_field}

                        if tags is not None:
                            sentence_tags =tags[delimiter[0]:delimiter[1]]
                            assert len(sentence_tags) == len(sentence_tokens)
                            fields["tags"] = SequenceLabelField(labels=sentence_tags, sequence_field=sentence_tokens_field)

                        if unary_marginals is not None:
                            sentence_unary_marginals = unary_marginals[delimiter[0]:delimiter[1]]
                            assert len(sentence_unary_marginals) == len(sentence_tokens)
                            fields['unary_marginals'] = ArrayField(sentence_unary_marginals)

                        if pairwise_marginals is not None:
                            sentence_pairwise_marginals = pairwise_marginals[delimiter[0]:delimiter[1]]
                            assert len(sentence_pairwise_marginals) == len(sentence_tokens)
                            fields['pairwise_marginals'] = ArrayField(sentence_pairwise_marginals)

                        if vote_mask is not None:
                            sentence_vote_mask = vote_mask[delimiter[0]:delimiter[1]]
                            assert len(sentence_vote_mask) == len(sentence_tokens)
                            fields["vote_mask"] = ArrayField(sentence_vote_mask)

                        yield Instance(fields)
        