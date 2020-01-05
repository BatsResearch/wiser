from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from typing import Iterator, List, Dict
import csv
import pickle

@DatasetReader.register('media')
class MediaDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, actor: str, tokens: List[Token], tags: List[str] = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        fields['actor_name'] = MetadataField(actor)

        return Instance(fields)

    def _read(self, file_path):

        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=",")
            tokens = []
            tags = []
            for row in reader:
                word, label = row

                if label == 'i-MOV':
                    label = 'I-MOV'
                if word == '\n':
                    continue
                if word == "*START-SENTENCE*":
                    tokens = []
                    tags = []
                elif word == "*END-SENTENCE*":
                    if len(tokens) > 1:
                        yield self.text_to_instance(label, tokens, tags)
                    tokens = []
                    tags = []
                elif word == "*START-ACTOR*" or word == "*END-ACTOR*":
                    continue
                else:
                    assert label in {'I-MOV', 'I-AWD', 'B-MOV', 'B-AWD', 'O'}
                    tokens.append(Token(word))
                    tags.append(label)
