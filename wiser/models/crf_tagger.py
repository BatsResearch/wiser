from typing import Dict, List, Any

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models import CrfTagger
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.conditional_random_field import allowed_transitions
import allennlp.nn.util as util
from wiser.modules import WiserConditionalRandomField


@Model.register("wiser_crf_tagger")
class WiserCrfTagger(CrfTagger):
    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder, **kwargs):
        super().__init__(vocab, text_field_embedder, encoder, **kwargs)

        # Gets the kwargs needs to initialize the WISER CRF. We skip some
        # configuration checks that are checked in the super constructor
        if kwargs.get('constrain_crf_decoding', None):
            labels = self.vocab.get_index_to_token_vocabulary(self.label_namespace)
            constraints = allowed_transitions(self.label_encoding, labels)
        else:
            constraints = None

        include_start_end_transitions = kwargs.get('include_start_end_transitions', True)

        # Replaces the CRF created by the super constructor with the WISER CRF
        self.crf = WiserConditionalRandomField(
            self.num_tags, constraints,
            include_start_end_transitions=include_start_end_transitions
        )

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Same signature as parent class's forward() method.

        Only difference is that loss is computed as the expected log likelihood
        using metadata, rather than tags.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if metadata is not None and 'WISER_LABELS' in metadata[0]:
            # Add negative log-likelihood as loss
            wiser_labels = [x['WISER_LABELS'] for x in metadata]
            ell = self.expected_log_likelihood(logits, wiser_labels, mask)
            output["loss"] = -ell

        if tags is not None:
            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())

        if metadata is not None and 'words' in metadata[0]:
            output["words"] = [x["words"] for x in metadata]
        return output
