from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter, WordTokenizer
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

        with open(file_path) as f:
            for line_num, line in enumerate(tqdm(f.readlines())):
                pieces = line.strip().split("\t")
                doc_id = pieces[0]
                title = pieces[1]

                # Puts a space before the closing period to match tokenization of abstract
                title = title[:-1] + " ."
                abstract = pieces[2]

                text = title + " " + abstract

                # Puts spaces around entity tags to ensure tokens are split correctly
                text = text.replace("<category=\"", " <category=\"")
                text = text.replace("\">", "\"> ")
                text = text.replace("</category>", " </category> ")

                initial_tokens = text.split()
                second_tokens = []

                # Processes the annotations to build up a string with tokens that
                # indicate starts and ends of spans
                for token in initial_tokens:
                    if token.startswith("<category="):
                        second_tokens.append(" /START/ ")
                    elif token == "</category>":
                        second_tokens.append(" /END/ ")
                    else:
                        second_tokens.append(token)

                # Creates new text to be parsed by Spacy
                text = " ".join(second_tokens)
                third_tokens = tokenizer.tokenize(text)

                # Iterates over all tokens and creates tag sequence
                tokens = []
                tags = []

                # 0 = no open span, 1 = next token is I, 2 = next token is B,
                # 3 = last token was end, so we don't yet know next token's label
                open_tag = 0
                for token in third_tokens:
                    # If the token is the start of a disease span, start a new tag sequence
                    if token.text == "/START/" and open_tag == 0:
                        open_tag = 1
                    elif token.text == "/START/" and open_tag == 3:
                        open_tag = 2
                    elif token.text == "/END/":
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

                yield self.text_to_instance(doc_id, tokens, tags)
