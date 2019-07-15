from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from tqdm.auto import tqdm
from typing import Iterator, List, Dict, Tuple
from xml.etree import ElementTree


class CDRDatasetReader(DatasetReader):
    """
    DatasetReader for CDR corpus available at
    https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/.

    This is an abstract class. Concrete subclasses need to implement _read by
    calling _cdr_read with a second argument that is the name of the entity type
    to include in the data set.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None, sentence_spans: List[Tuple[int, int]]=None) -> Instance:

        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        if sentence_spans:
            fields['sentence_spans'] = MetadataField(sentence_spans)

        return Instance(fields)

    # Returns true if current entity is different from previous entity
    def _entity_is_different_from_previous_entity(self, ix, current_entity, i_tokens):
        return ix > 0 and tags[ix-1] == 'I' and current_entity > 0
                and i_tokens[current_entity][1] != i_tokens[current_entity-1][1]

    def _is_i_tag(self, tokens, ix, current_entity, i_tokens):
        return current_entity < len(i_tokens)
                and str(tokens[ix]) == i_tokens[current_entity][0]

    # Reformats text to ensure sound tokenizing
    def _reformat_text(self, text):
        replace_dict = {'.': ' . ', '-': ' - ', '/': ' / ', '=': ' = '}
        for key, value in replace_dict.items():
            text = text.replace(key, value)
        return text


    def _read(self, file_path: str) -> Iterator[Instance]:
        raise NotImplementedError

    def _cdr_read(self, file_path: str, entity_type: str) -> Iterator[Instance]:
        splitter = SpacyWordSplitter('en_core_web_sm', True, True, True)
        tokenizer = WordTokenizer(word_splitter=splitter)
        splitter = SpacySentenceSplitter()
        root = ElementTree.parse(file_path).getroot()
        xml_docs = root.findall("./document")
        for xml_doc in tqdm(xml_docs):
            xml_title = xml_doc.find("passage[infon='title']")
            xml_abstract = xml_doc.find("passage[infon='abstract']")

            doc_name = xml_doc.find('id').text
            title = xml_title.find('text').text
            abstract = xml_abstract.find('text').text
            text = title + " " + abstract

            # Collects all annotations so that they can be sorted and processed
            annotations = []
            for xml in (xml_title, xml_abstract):
                xml_annotations = xml.findall("annotation[infon='" + entity_type + "']")
                for annotation in xml_annotations:
                    # Skips IndividualMentions, since they are subsumed by
                    # CompositeMentions
                    keep = True
                    for infon in annotation.findall('infon'):
                        if infon.text == 'IndividualMention':
                            keep = False
                    if keep:
                        annotations.append(annotation)

            # Sorts the annotations by start character
            annotations.sort(key=lambda x: int(x.find('location').get('offset')))

            i_tokens = []
            for span_id, annotation in enumerate(annotations):
                start = int(annotation.find('location').get('offset'))
                end = start + int(annotation.find('location').get('length'))
                span_tokens = tokenizer.tokenize(self._reformat_text(text[start:end]))
                i_tokens += [(str(token), span_id) for token in span_tokens]

            # # Splits the text into sentences, and tokenizes each sentence
            sentences = splitter.split_sentences(self._reformat_text(text))
            sentence_spans = []
            sentence_start = 0
            sentence_end = 0
            tokens = []
            for sentence in sentences:
                sentence_tokens = tokenizer.tokenize(sentence)

                sentence_end = sentence_start + len(sentence_tokens) - 1
                sentence_spans.append((sentence_start, sentence_end))
                sentence_start = sentence_end + 1
                tokens += sentence_tokens

            tags = []
            current_entity = 0
            for ix, token in enumerate(tokens):
                if self._is_i_tag(self, tokens, ix, current_entity, i_tokens):
                    if self._entity_is_different_from_previous_entity(current_entity, i_tokens):
                        tags.append('B')
                    else:
                        tags.append('I')
                    current_entity += 1
                else:
                    tags.append('O')

            assert current_entity == len(i_tokens)

            yield self.text_to_instance(tokens, tags, sentence_spans)

@DatasetReader.register('cdr_disease')
class CDRDiseaseDatasetReader(CDRDatasetReader):
    """
    DatasetReader for CDR Disease corpus.
    """
    def _read(self, file_path: str) -> Iterator[Instance]:
        return self._cdr_read(file_path, 'Disease')


@DatasetReader.register('cdr_chemical')
class CDRChemicalDatasetReader(CDRDatasetReader):
    """
    DatasetReader for CDR Chemical corpus.
    """
    def _read(self, file_path: str) -> Iterator[Instance]:
        return self._cdr_read(file_path, 'Chemical')
