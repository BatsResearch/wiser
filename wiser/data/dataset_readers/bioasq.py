# from allennlp.data import Instance
# from allennlp.data.dataset_readers import DatasetReader
# from allennlp.data.fields import TextField, SequenceLabelField
# from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
# from allennlp.data.tokenizers import Token
# from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter, WordTokenizer
# from tqdm.auto import tqdm
# from typing import Iterator, List, Dict
# from xml.etree import ElementTree
# import json
# import pdb
#
# @DatasetReader.register('bioasq')
# class BioASQDatasetReader(DatasetReader):
#     """
#     DatasetReader for BioASQ corpus.
#     """
#
#     def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
#         super().__init__(lazy=False)
#         self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
#
#     def text_to_instance(self, doc_id: str, tokens: List[Token], tags: List[str] = None) -> Instance:
#         tokens_field = TextField(tokens, self.token_indexers)
#         fields = {"tokens": tokens_field}
#
#         return Instance(fields)
#
#     def _read(self, file_path: str) -> Iterator[Instance]:
#
#         # Keys: title + abstractText
#         splitter = SpacyWordSplitter('en_core_web_sm', True, True, True)
#         tokenizer = WordTokenizer(word_splitter=splitter)
#         with open(file_path, 'r') as f:
#             json_docs = json.load(f)
#
#         for article in json_docs['documents']:
#             doc_name = article['pmid']
#             title = article['title']
#             abstract = article['abstractText']
#             text = title + " " + abstract
#
#             tokens = tokenizer.tokenize(text)
#
#             yield self.text_to_instance(doc_name, tokens)
