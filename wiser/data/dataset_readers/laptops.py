from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter, WordTokenizer
from tqdm.auto import tqdm
from typing import Iterator, List, Dict
from xml.etree import ElementTree


@DatasetReader.register('laptops')
class LaptopsDatasetReader(DatasetReader):
    """
    DatasetReader for Laptop Reviews corpus available at
    http://alt.qcri.org/semeval2014/task4/.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, doc_id: str, tokens: List[Token], tags: List[str] = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        splitter = SpacyWordSplitter('en_core_web_sm', True, True, True)
        tokenizer = WordTokenizer(word_splitter=splitter)
        root = ElementTree.parse(file_path).getroot()
        xml_sents = root.findall("./sentence")

        for xml_sent in tqdm(xml_sents):
            text = xml_sent.find("text").text
            annotations = xml_sent.find('aspectTerms')
            if annotations is not None:
                annotations = annotations.findall("aspectTerm")
            else:
                annotations = []

            # Sorts the annotations by start character
            annotations.sort(key=lambda x: int(x.get('from')))

            # Tokenizes the sentence
            tokens = tokenizer.tokenize(text)

            # Assigns tags based on annotations
            tags = []
            next = 0
            current = None
            for token in tokens:
                # Checks if the next annotation begins somewhere in this token
                start_entity = next < len(annotations)
                start_entity = start_entity and token.idx <= int(annotations[next].get('from'))
                start_entity = start_entity and token.idx + len(token.text) > int(annotations[next].get('from'))

                if start_entity:
                    tags.append('I' if current is None else 'B')
                    current = annotations[next]
                    next += 1
                elif current is not None:
                    if token.idx < int(current.get('to')):
                        tags.append('I')
                    else:
                        tags.append('O')
                        current = None
                else:
                    tags.append('O')

            yield self.text_to_instance(xml_sent.get('id'), tokens, tags)
