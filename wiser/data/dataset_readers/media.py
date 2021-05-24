from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from typing import List, Dict
import csv

@DatasetReader.register('media')
class MediaDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None, actor: str = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        if actor:
            fields['actor'] = MetadataField(actor)

        return Instance(fields)

    def _read(self, file_path):

        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=",")
            tokens = []
            tags = []
            for row in reader:
                word, label = row

                if word == '\n':
                    continue
                if word == "*START-SENTENCE*":
                    tokens = []
                    tags = []
                elif word == "*END-SENTENCE*":
                    if len(tokens) > 1:
                        yield self.text_to_instance(tokens, tags, label)
                    tokens = []
                    tags = []
                elif word == "*START-ACTOR*" or word == "*END-ACTOR*":
                    continue
                else:
                    if label not in {'I-PERF', 'I-AWD', 'B-PERF', 'B-AWD', 'O'}:
                        raise RuntimeError('Label %s is not a valid tag' % label)
                    tokens.append(Token(word))
                    tags.append(label)
