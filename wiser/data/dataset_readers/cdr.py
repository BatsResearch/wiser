# from allennlp.data import Instance
# from allennlp.data.dataset_readers import DatasetReader
# from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
# from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
# from allennlp.data.tokenizers import Token
# import spacy
# from spacy.tokenizer import Tokenizer
# from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
# from tqdm.auto import tqdm
# from typing import Iterator, List, Dict, Tuple
# from xml.etree import ElementTree
#
#
# class CDRDatasetReader(DatasetReader):
#     """
#     DatasetReader for CDR corpus available at
#     https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/.
#
#     This is an abstract class. Concrete subclasses need to implement _read by
#     calling _cdr_read with a second argument that is the name of the entity type
#     to include in the data set.
#     """
#     def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, use_regex: bool = True) -> None:
#         super().__init__(lazy=False)
#         self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
#
#         self.nlp = spacy.load('en_core_web_sm')
#
#         if use_regex:
#             infix_re = compile_infix_regex(self.nlp.Defaults.infixes + tuple(r'-') + tuple(r'[/+=\(\)\[\]]'))
#             prefix_re = compile_prefix_regex(self.nlp.Defaults.prefixes + tuple(r'[\'\(\[]'))
#             suffix_re = compile_suffix_regex(self.nlp.Defaults.suffixes + tuple(r'[\.\+\)\]]'))
#
#             self.nlp.tokenizer =  Tokenizer(self.nlp.vocab, prefix_search=prefix_re.search,
#                                         suffix_search=suffix_re.search,
#                                         infix_finditer=infix_re.finditer,
#                                         token_match=self.nlp.tokenizer.token_match)
#
#     def get_tokenizer(self):
#         return self.nlp.tokenizer
#
#     def text_to_instance(self, tokens: List[Token], tags: List[str] = None, sentence_spans: List[Tuple[int, int]]=None) -> Instance:
#
#         tokens_field = TextField(tokens, self.token_indexers)
#         fields = {"tokens": tokens_field}
#
#         if tags:
#             tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
#             fields["tags"] = tags_field
#
#         if sentence_spans:
#             fields['sentence_spans'] = MetadataField(sentence_spans)
#
#         return Instance(fields)
#
#     def _read(self, file_path: str) -> Iterator[Instance]:
#         raise NotImplementedError
#
#     def _cdr_read(self, file_path: str, entity_types: List[str]) -> Iterator[Instance]:
#         root = ElementTree.parse(file_path).getroot()
#         xml_docs = root.findall("./document")
#         for xml_doc in tqdm(xml_docs):
#             xml_title = xml_doc.find("passage[infon='title']")
#             xml_abstract = xml_doc.find("passage[infon='abstract']")
#
#             title = xml_title.find('text').text
#             abstract = xml_abstract.find('text').text
#             raw_text = title + " " + abstract
#
#             doc = self.nlp(raw_text)
#             sentences = [sent for sent in doc.sents]
#             tokens = [token for sentence in sentences for token in sentence]
#             sentence_spans = [(sent.start, sent.end) for sent in sentences]
#
#             tags = ['O'] * len(tokens)
#             for entity_type in entity_types:
#
#                 i_tag = 'I-%s' % (entity_type)
#                 b_tag = 'B-%s' % (entity_type)
#
#                 xml_annotations = []
#                 for xml in (xml_title, xml_abstract):
#                     annotations = xml.findall("annotation[infon='" + entity_type + "']")
#                     xml_annotations += annotations
#
#                 xml_annotations.sort(key=lambda x: int(x.find('location').get('offset')))
#
#                 for annotation in xml_annotations:
#                     # Skips IndividualMentions, since they are subsumed by
#                     # CompositeMentions
#                     if 'IndividualMention' not in [a.text for a in annotation.findall('infon')]:
#                         start = int(annotation.find('location').get('offset'))
#                         end = start + int(annotation.find('location').get('length'))
#                         entity_span = doc.char_span(start, end)
#
#                         if entity_span is None:
#                             start = raw_text.rfind(' ', 0, start) + 1
#                             end = raw_text.find(' ', end)
#                             start = start if start != -1 else 0
#                             end = end if end != -1 else len(raw_text)
#
#                             entity_span = doc.char_span(start, end)
#
#                         entity_start = entity_span.start
#                         entity_end = entity_span.end
#
#                         if entity_start > 0 and tags[entity_start-1] != 'O':
#                             tags[entity_start] = b_tag
#                         else:
#                             tags[entity_start] = i_tag
#
#                         tags[entity_start+1:entity_end] = [i_tag] * (len(entity_span)-1)
#
#             tokens = [Token(token.text,
#                 token.idx,
#                 token.lemma_,
#                 token.pos_,
#                 token.tag_,
#                 token.dep_,
#                 token.ent_type_) for sentence in sentences for token in sentence]
#
#             yield self.text_to_instance(tokens, tags, sentence_spans)
#
# @DatasetReader.register('cdr')
# class CDRCombinedDatasetReader(CDRDatasetReader):
#     """
#     DatasetReader for CDR Disease + Chemical corpus.
#     """
#     def _read(self, file_path: str) -> Iterator[Instance]:
#         return self._cdr_read(file_path, ['Disease', 'Chemical'])
#
# @DatasetReader.register('cdr_disease')
# class CDRDiseaseDatasetReader(CDRDatasetReader):
#     """
#     DatasetReader for CDR Disease corpus.
#     """
#     def _read(self, file_path: str) -> Iterator[Instance]:
#         return self._cdr_read(file_path, ['Disease'])
#
# @DatasetReader.register('cdr_chemical')
# class CDRChemicalDatasetReader(CDRDatasetReader):
#     """
#     DatasetReader for CDR Chemical corpus.
#     """
#     def _read(self, file_path: str) -> Iterator[Instance]:
#         return self._cdr_read(file_path, ['Chemical'])
#
