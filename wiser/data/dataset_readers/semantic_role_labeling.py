import logging
from typing import Dict, List, Iterable, Tuple, Any

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers import SrlReader
import spacy
import numpy as np
import pdb
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("srl-iob1")
class SrlReaderIOB1(SrlReader):

    def __init__(self,
             token_indexers: Dict[str, TokenIndexer] = None,
             domain_identifier: str = None,
             lazy: bool = False,
             bert_model_name: str = None,
             used_tags: set = None,
             dependency_parse: bool = False) -> None:

        super().__init__(token_indexers=token_indexers,  domain_identifier=domain_identifier,
                        lazy=lazy)
        self.used_tags = used_tags
        self.dependency_parse = dependency_parse

        if dependency_parse:
            self.nlp = spacy.load('en_core_web_sm',
                disable=["tagger", "ner", 'textcat', '...'])

    @overrides
    def _read(self, file_path: str):

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):

            pos_tags = [t for t in sentence.pos_tags]

            tokens = [Token(t, None, None, pos_tags[i])
                    for i, t in enumerate(sentence.words)]

            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    verb_indices = np.where(np.array(verb_indicator)==1)[0]

                    if len(verb_indices) > 0:
                        verb_index = int(verb_indices[0])
                        verb = tokens[verb_index]
                    else:
                        verb_index = -1
                        verb = ''

                    for i, tag in enumerate(tags):
                        if tag[0] == 'B':
                            tags[i] = tags[i].replace('B', 'I', 1)
                        if self.used_tags is not None and tags[i] not in self.used_tags:
                            tags[i] = 'O'

                    instance = self.text_to_instance([verb] + tokens, [0] + verb_indicator, ['O'] + tags)

                    if self.dependency_parse:
                        doc = self.nlp(' '.join(sentence.words))
                        instance.add_field('dependency', MetadataField(doc))

                    instance.add_field('verb_index', IndexField(verb_index, instance['tokens']))
                    yield instance
