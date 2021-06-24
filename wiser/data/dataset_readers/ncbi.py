from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from tqdm.auto import tqdm
from typing import Iterator, List, Dict


@DatasetReader.register('ncbi-disease')
class NCBIDiseaseDatasetReader(DatasetReader):
    """
    DatasetReader for NCBI Disease corpus available at
    https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/.

    Note that the corpus is available in two formats. This DatasetReader is for
    the "mention level" corpus available at
    https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, use_regex: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.nlp = spacy.load('en_core_web_sm')

        if use_regex:
            infix_re = compile_infix_regex(self.nlp.Defaults.infixes + list(r'-') + list(r'[/+=\(\)\[\]]'))
            prefix_re = compile_prefix_regex(self.nlp.Defaults.prefixes + list(r'[\'\(\[]'))
            suffix_re = compile_suffix_regex(self.nlp.Defaults.suffixes + list(r'[\.\+\)\]]'))

            self.nlp.tokenizer = Tokenizer(
                self.nlp.vocab,
                prefix_search=prefix_re.search,
                suffix_search=suffix_re.search,
                infix_finditer=infix_re.finditer,
                token_match=self.nlp.tokenizer.token_match)

    def text_to_instance(self, doc_id: str, tokens: List[Token], tags: List[str] = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            title, abstract = None, None
            annotations = []
            for line in tqdm(f.readlines()):
                if line.strip() == '':
                    if title is not None:
                        doc = self.nlp(title + " " + abstract)

                        # Sorts the annotations by start character
                        annotations.sort(key=lambda x: int(x[0]))

                        tokens = [Token(text=token.text,
                                        idx=token.idx,
                                        lemma_=token.lemma_,
                                        pos_=token.pos_,
                                        tag_=token.tag_,
                                        dep_=token.dep_,
                                        ent_type_=token.ent_type_) for token in doc]

                        # Assigns tags based on annotations
                        tags = []
                        next = 0
                        current = None
                        for token in tokens:
                            # Checks if the next annotation begins somewhere in this token
                            start_entity = next < len(annotations)
                            start_entity = start_entity and token.idx <= annotations[next][0]
                            start_entity = start_entity and token.idx + len(token.text) > int(annotations[next][0])

                            if start_entity:
                                tags.append('I' if current is None else 'B')
                                current = annotations[next]
                                next += 1
                            elif current is not None:
                                if token.idx < int(current[1]):
                                    tags.append('I')
                                else:
                                    tags.append('O')
                                    current = None
                            else:
                                tags.append('O')

                        yield self.text_to_instance(doc_id, tokens, tags)
                        title, abstract, doc = None, None, None
                        annotations = []
                    continue

                if title is None:
                    pieces = line.strip().split("|", 3)
                    doc_id = pieces[0]
                    title = pieces[2]
                    continue
                elif abstract is None:
                    abstract = line.strip().split("|", 3)[2]
                    continue
                else:
                    pieces = line.strip().split()
                    annotations.append((int(pieces[1]), int(pieces[2])))
