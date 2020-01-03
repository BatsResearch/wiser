from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from typing import Iterator, List, Dict
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

    def _read(self, file_path: str) -> Iterator[Instance]:
        actor_dict = pickle.load(open(file_path, 'rb'))

        for actor, sentences in actor_dict.items():
            for sent in sentences:
                tokens = [Token(t[0]) for t in sent]
                tags = None
                if len(sent[0]) == 2:
                    tags = [t[1] for t in sent]

                yield self.text_to_instance(actor, tokens, tags)
