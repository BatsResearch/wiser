import itertools
import os
from .lf import DictionaryMatcher


class UMLSMatcher(DictionaryMatcher):
    def __init__(
            self, name, path, sem_types, additional_stop_words=None,
            max_tokens=4, i_label="I", abs_label="ABS"):
        # Collects concept IDs of the requested semtantic type(s)
        CUIs = set()
        sem_type_file = os.path.join(path, 'META', 'MRSTY.RRF')
        with open(sem_type_file, 'r') as f:
            for line in f.readlines():
                data = line.strip().split('|')
                if data[3] in sem_types:
                    CUIs.add(data[0])

        # Collects all English terms corresponding to the collected concepts
        terms = set()
        name_file = os.path.join(path, 'META', 'MRXNS_ENG.RRF')
        with open(name_file, 'r') as f:
            for line in f.readlines():
                data = line.strip().split('|')
                if data[2] in CUIs and len(data[1].split(' ')) <= max_tokens:
                    terms.add(data[1])

        # Removes stop words that are too common from terms
        self._filter_stop_words(terms, additional_stop_words)

        # Tokenizes the UMLS terms
        terms = [term.split(' ') for term in terms]

        # Creates reordered copies of terms for common groups,
        # e.g., "cancer ovarian" -> "ovarian cancer"
        self._expand_terms(terms)

        # Finished intializing the labeling function
        super(UMLSMatcher, self).__init__(
            name, terms, match_lemmas=True, i_label=i_label, abs_label=abs_label)

    def _normalize_instance_tokens(self, tokens, lemmas=False):
        if lemmas:
            normalized_tokens = [token.lemma_ for token in tokens]
        else:
            normalized_tokens = [token.text.lower() for token in tokens]
        return normalized_tokens

    def _expand_terms(self, terms):
        new_terms = set()
        for term in terms:
            for new_term in itertools.permutations(term):
                new_terms.add(" ".join(new_term))
        terms.extend(term.split(' ') for term in new_terms)

    def _filter_stop_words(self, terms, additional_stop_words):
        """
        We remove terms that appear in UMLS that are unlikely to be terms of
        interest (by themselves) for NER. Users who want to change this behavior
        can override this method.

        :param terms:
        :param additional_stop_words:
        """
        stop_words = {
            "a", "active", "age", "agent", "advantage", "aim", "air", "al", "alert", "all", "an", "animal", "as", "at", "atm",
            "b", "balance", "base", "basic", "basis", "be", "blockade", "bp", "but",
            "c", "can", "color", "combination", "condition", "contrast", "control", "counter", "culture",
            "d", "damage", "date", "de", "defect", "deletion", "direct", "disease", "disorder", "doctor", "drug", "duplication", "duration",
            "e", "end", "et", "exposure",
            "f", "family", "finding", "food",
            "g",
            "h", "he", "her", "hers", "hg", "hh", "his",
            "i", "ii", "if", "in", "inactive", "ingredient", "injection", "is",
            "j",
            "k",
            "l", "label", "level", "light",
            "m", "man", "mass", "matrix", "mediate", "medication", "men", "messenger", "met", "ml", "mg", "mm", "mutant", "mutation",
            "n", "neuroleptic", "no", "nonsense",
            "o", "opioid", "or", "oral", "other",
            "p", "perform", "personality", "placebo", "plasma", "pool", "prevent", "program", "prompt", "prove", "purpose", "psychotropic",
            "q",
            "r", "receptor", "relief", "rise", "rose", "rr",
            "s", "same", "se", "sham", "smell", "so", "solution", "solid", "stain", "support", "syndrome",
            "t", "ten", "therapeutic", "tilt", "to", "tonic", "transition", "tricyclic",
            "u", "unknown", "us",
            "v", "various", "vessel",
            "w", "was", "water",
            "x",
            "y",
            "z"
        }

        if additional_stop_words is not None:
            for stop_word in additional_stop_words:
                stop_words.add(stop_word)

        for stop_word in stop_words:
            if stop_word in terms:
                terms.remove(stop_word)
