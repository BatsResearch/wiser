from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from typing import Iterator, Dict
from allennlp.data.fields import ArrayField, TextField, SequenceLabelField
import pickle
import numpy as np

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

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for instance in data:
            tokens = [token for token in instance['tokens']]
            instance.add_field('tokens', TextField(tokens, self.token_indexers))
            yield instance
