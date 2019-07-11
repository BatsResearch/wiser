import json
import os
from typing import Dict, Iterator, List
from random import shuffle

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_tokenizer import (SpacyWordSplitter,
                                                     WordTokenizer)


@DatasetReader.register('ppdr-dataset')
class ProductPageDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, majority_vote=True, ignore_punctuation=True, shuffle_data=False, max_token_length=500) -> None:
        super().__init__(lazy=True)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.ignore_punctuation = ignore_punctuation
        self.punctuations = {'[', '(', '.', ','}
        self.count_vocab_items = self.token_indexers
        self.majority_vote = majority_vote
        self.max_token_length = max_token_length
        self.shuffle_data = shuffle_data

    @staticmethod
    def get_annotations(annotation_file: str):

        annotation = open(annotation_file, 'rb') # Merged annotation result from MTurk
        annotation = annotation.read().decode(errors='ignore')

        name_offsets, feature_offsets = annotation.replace("'", '"').split('\n') #JS quotes and Python quotes are different. Replace them.
        
        name_offsets = json.loads(name_offsets)['indices'] if name_offsets != "NOT_DONE_YET" else []
        feature_offsets = json.loads(feature_offsets)['indices'] if feature_offsets != "NOT_DONE_YET" else []

        return name_offsets, feature_offsets

    @staticmethod
    def is_annotated(offsets: List, current_range: set):
        for offset in offsets:
            start, end = offset
            new_range = set(range(start, end+1))
            intersection = current_range.intersection(new_range)
            if intersection:
                return True, new_range
        return False, None

    @staticmethod
    def is_annotated_majority_vote(name_offsets: List, feature_offsets: List, current_range: set):
        '''Checks 3 annotations and assigns labels to tokens based on a majority vote'''
        
        name_votes = 0
        feature_votes = 0

        name_range = None
        feature_range = None

        for offsets in name_offsets:
            for offset in offsets:
                start, end = offset
                new_range = set(range(start, end+1))
                intersection = current_range.intersection(new_range)
                if intersection:
                    name_votes += 1
                    name_range = new_range
                    break

        # Token is tagged as 'name'
        if name_votes > 1:
            return True, name_range, False, feature_range 

        for offsets in feature_offsets:
            for offset in offsets:
                start, end = offset
                new_range = set(range(start, end+1))
                intersection = current_range.intersection(new_range)
                if intersection:
                    feature_votes += 1      
                    feature_range = new_range
                    break
        
        # Token is tagged as 'feature'
        if feature_votes > 1:
            return False, name_range, True, feature_range 

        # Token is tagged as 'other'
        return False, name_range, False, feature_range



    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        return Instance(fields)

    @staticmethod
    def download_data(data_url: str):

        dataset_name = data_url.split('/')[-1].split('.')[0]

        if not os.path.isdir('./data/'+dataset_name): # Do not download if already exists
            import urllib.request, tarfile
            
            print("Downloading dataset...")
            file_tmp = urllib.request.urlretrieve(data_url, filename=None)
            tar = tarfile.open(file_tmp[0])
            tar.extractall('./data/')

        data_path = './data/' + dataset_name + '/text_files/'
        annotation_path = './data/' + dataset_name + '/annotations/'

        return data_path, annotation_path


    def _read(self, data_url: str) -> Iterator[Instance]:

        splitter  = SpacyWordSplitter('en_core_web_sm', True, True, True)
        tokenizer = WordTokenizer(word_splitter=splitter)

        data_path, annotation_path = self.download_data(data_url)
        dataset = os.listdir(data_path)
        
        if self.shuffle_data:
            shuffle(dataset)

        for index, _ in enumerate(dataset):

            current_text_file = dataset[index]

            if self.majority_vote:
                # Red annotations ( majority vote )
                name_offsets_0, feature_offsets_0 = self.get_annotations(annotation_file=annotation_path + '0_' + current_text_file)
                name_offsets_1, feature_offsets_1 = self.get_annotations(annotation_file=annotation_path + '1_' + current_text_file)
                name_offsets_2, feature_offsets_2 = self.get_annotations(annotation_file=annotation_path + '2_' + current_text_file)
            else:
                # Read annotations ( union merge )
                name_offsets, feature_offsets = self.get_annotations(annotation_file=annotation_path + '3_' + current_text_file)

            data = open(data_path+current_text_file, 'rb') # Extracted text from html page
            data = data.read().decode(errors='ignore')

            doc = tokenizer.tokenize(data)

            if len(doc) <= self.max_token_length:
                number_of_documents = 1
            else:
                number_of_documents = (len(doc) // self.max_token_length) + 1

            for doc_id in range(number_of_documents):

                doc_ = doc[doc_id*self.max_token_length:(doc_id+1)*self.max_token_length]

                tokens = []
                tags   = []

                previous = ["other", None] # Label, Range

                for token in doc_:
                    text = token.text
                    start_offset = token.idx
                    end_offset = start_offset + len(text)
                    current_range = set(range(start_offset, end_offset+1))

                    if self.ignore_punctuation:
                        if previous[0] != "other" and (text in self.punctuations):
                            tag_ = "O"
                            previous = ["other", None]
                            tokens.append(Token(token.text, token.idx, token.lemma_, token.pos_, token.tag_, token.dep_, token.ent_type_))
                            tags.append(tag_)
                            continue

                    if self.majority_vote:
                        is_name_annotated, name_range, is_feature_annotated, feature_range = self.is_annotated_majority_vote(name_offsets=[name_offsets_0, name_offsets_1, name_offsets_2], feature_offsets=[feature_offsets_0, feature_offsets_1, feature_offsets_2], current_range=current_range)
                    else:
                        is_name_annotated, name_range = self.is_annotated(offsets=name_offsets, current_range=current_range)
                        is_feature_annotated, feature_range = self.is_annotated(offsets=feature_offsets, current_range=current_range)

                    if is_name_annotated: # "I-NAME" or "B-NAME"
                        if previous[0] == "other":
                            tag_ = "I-NAME"
                        else:
                            previous_name, previous_range = previous
                            if previous_name == "name": # Check if this is a new highlight or continuation of the old one?
                                if previous_range == name_range:
                                    tag_ = "I-NAME"
                                else:
                                    tag_ = "B-NAME"
                            else: #Previous selection was a feature
                                tag_ = "I-NAME"
                        previous = ["name", name_range]
                    elif is_feature_annotated: # "I-FEAT" or "B-FEAT"
                        if previous[0] == "other":
                            tag_ = "I-FEAT"
                        else:
                            previous_name, previous_range = previous
                            if previous_name == "feature": # Check if this is a new highlight or continuation of the old one?
                                if previous_range == feature_range:
                                    tag_ = "I-FEAT"
                                else:
                                    tag_ = "B-FEAT"
                            else: #Previous selection was a name
                                tag_ = "I-FEAT"
                        previous = ["feature", feature_range]
                    else:
                        tag_ = "O"
                        previous = ["other", None]
                        
                    tokens.append(Token(token.text,
                                        token.idx,
                                        token.lemma_,
                                        token.pos_,
                                        token.tag_,
                                        token.dep_,
                                        token.ent_type_))
                    tags.append(tag_)

                yield self.text_to_instance(tokens, tags)
