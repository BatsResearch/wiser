from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter, WordTokenizer
from tqdm.auto import tqdm
from typing import Iterator, List, Dict
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

    def text_to_instance(self, doc_id: str, tokens: List[Token], tags: List[str] = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        raise NotImplementedError

    def _cdr_read(self, file_path: str, entity_type: str) -> Iterator[Instance]:
        splitter = SpacyWordSplitter('en_core_web_sm', True, True, True)
        tokenizer = WordTokenizer(word_splitter=splitter)
        root = ElementTree.parse(file_path).getroot()
        xml_docs = root.findall("./document")

        for xml_doc in tqdm(xml_docs):
            xml_title = xml_doc.find("passage[infon='title']")
            xml_abstract = xml_doc.find("passage[infon='abstract']")

            doc_name = xml_doc.find('id').text
            title = xml_title.find('text').text
            abstract = xml_abstract.find('text').text
            raw_text = title + " " + abstract

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

            # Processes the annotations to build up a string with tokens that
            # indicate starts and ends of spans
            text = ""
            last_end = 0
            for annotation in annotations:
                start = int(annotation.find('location').get('offset'))
                end = start + int(annotation.find('location').get('length'))

                # We use " /START/ " and " /END/ " so that they are passed
                # through Spacy's tokenizer without being split
                text = text + raw_text[last_end:start] + " /START/ " + \
                    raw_text[start:end] + " /END/ "
                last_end = end

            # Puts the last end on
            text = text + raw_text[last_end:]

            initial_tokens = tokenizer.tokenize(text)
            tokens = []
            tags = []

            # Iterates over all tokens and creates tag sequence

            # 0 = no open span, 1 = next token is I, 2 = next token is B,
            # 3 = last token was end, so we don't yet know next token's label
            open_tag = 0
            for token in initial_tokens:
                # If the token is the start of a disease span, start a new tag sequence
                if str(token) == "/START/" and open_tag == 0:
                    open_tag = 1
                elif str(token) == "/START/" and open_tag == 3:
                    open_tag = 2
                elif str(token) == "/END/":
                    open_tag = 3
                elif open_tag == 0:
                    tokens.append(token)
                    tags.append("O")
                elif open_tag == 1:
                    tokens.append(token)
                    tags.append("I")
                elif open_tag == 2:
                    tokens.append(token)
                    tags.append("B")
                    open_tag = 1
                elif open_tag == 3:
                    tokens.append(token)
                    tags.append("O")
                    open_tag = 0
                else:
                    raise RuntimeError("Unexpected state.")

            # We remove the indices because they are incorrect due to the
            # removal of the /START/ and /END/ tags
            tokens = [Token(token.text,
                            None,
                            token.lemma_,
                            token.pos_,
                            token.tag_,
                            token.dep_,
                            token.ent_type_) for token in tokens]

            yield self.text_to_instance(doc_name, tokens, tags)


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
