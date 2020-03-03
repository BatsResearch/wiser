from tqdm.auto import tqdm


def remove_rule(data, name):
    """
    Removes a tagging or linking rule from a given dataset
    """

    for instance in data:
        if name in instance['WISER_LABELS']:
            del instance['WISER_LABELS'][name]
        if name in instance['WISER_LINKS']:
            del instance['WISER_LINKS'][name]


class TaggingRule:
    def apply(self, instances):
        for instance in tqdm(instances):
            # Initializes metadata field
            if self._get_metadata_field() not in instance:
                instance.add_field(self._get_metadata_field(), {})

            # Labels the instance
            labels = self.apply_instance(instance)

            # Stores the labels in the instance
            instance[self._get_metadata_field()][self._get_tr_name()] = labels

    def apply_instance(self, instance):
        raise NotImplementedError

    def _get_metadata_field(self):
        return "WISER_LABELS"

    def _get_tr_name(self):
        return type(self).__name__


class LinkingRule(TaggingRule):
    def apply_instance(self, instance):
        raise NotImplementedError

    def _get_metadata_field(self):
        return "WISER_LINKS"


class DictionaryMatcher(TaggingRule):
    def __init__(self, name, terms, uncased=False, match_lemmas=False, i_label="I", abs_label="ABS"):
        self.name = name
        self.uncased = uncased
        self.match_lemmas = match_lemmas
        self.i_label = i_label
        self.abs_label = abs_label

        self._load_terms(terms)

    def apply_instance(self, instance):
        tokens = self._normalize_instance_tokens(instance['tokens'])
        labels = [self.abs_label] * len(instance['tokens'])

        # Checks whether any terms in the dictionary appear in the instance
        i = 0
        while i < len(tokens):
            if tokens[i] in self.term_dict:
                candidates = self.term_dict[tokens[i]]
                for c in candidates:
                    # Checks whether normalized AllenNLP tokens equal the list
                    # of string tokens defining the term in the dictionary
                    if i + len(c) <= len(tokens):
                        equal = True
                        for j in range(len(c)):
                            if tokens[i + j] != c[j]:
                                equal = False
                                break

                        # If tokens match, labels the instance tokens
                        if equal:
                            for j in range(i, i + len(c)):
                                labels[j] = self.i_label
                            i = i + len(c) - 1
                            break
            i += 1

        # Additionally checks lemmas if requested. This will not overwrite
        # existing votes
        if self.match_lemmas:
            tokens = self._normalize_instance_tokens(instance['tokens'], lemmas=True)
            i = 0
            while i < len(tokens):
                if tokens[i] in self.term_dict:
                    candidates = self.term_dict[tokens[i]]
                    for c in candidates:
                        # Checks whether normalized AllenNLP tokens equal the list
                        # of string tokens defining the term in the dictionary
                        if i + len(c) <= len(tokens):
                            equal = True
                            for j in range(len(c)):
                                if tokens[i + j] != c[j] or labels[i + j] != self.abs_label:
                                    equal = False
                                    break

                            # If tokens match, labels the instance tokens using map
                            if equal:
                                for j in range(i, i + len(c)):
                                    labels[j] = self.i_label
                                i = i + len(c) - 1
                                break
                i += 1

        return labels

    def _get_tr_name(self):
        return self.name

    def _normalize_instance_tokens(self, tokens, lemmas=False):
        if lemmas:
            normalized_tokens = [token.lemma_ for token in tokens]
        else:
            normalized_tokens = [token.text for token in tokens]

        if self.uncased:
            normalized_tokens = [token.lower() for token in normalized_tokens]

        return normalized_tokens

    def _normalize_terms(self, tokens):
        if self.uncased:
            return [token.lower() for token in tokens]
        return tokens

    def _load_terms(self, terms):
        self.term_dict = {}
        for term in terms:
            normalized_term = self._normalize_terms(term)

            if normalized_term[0] not in self.term_dict:
                self.term_dict[normalized_term[0]] = []

            self.term_dict[normalized_term[0]].append(normalized_term)

        # Sorts the terms in decreasing order so that we match the longest first
        for first_token in self.term_dict.keys():
            to_sort = self.term_dict[first_token]
            self.term_dict[first_token] = sorted(
                to_sort, reverse=True, key=lambda x: len(x))
