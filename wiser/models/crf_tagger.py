
from allennlp.models import CrfTagger
from typing import Dict, Optional, List, Any
from overrides import overrides
import torch
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from wiser.modules.conditional_random_field import WiserConditionalRandomField
import numpy as np
import pdb
from torch.nn.functional import log_softmax, softmax
from torch.nn import MSELoss, KLDivLoss, NLLLoss, MSELoss, BCEWithLogitsLoss, CrossEntropyLoss


# Computes CE-Loss with respect to one-hot vector corresponding to the tag with highest probability
def one_hot_cross_entropy_loss(logits, unary_marginals, num_tags):
    loss = CrossEntropyLoss()
    return loss(logits.view(-1, num_tags), unary_marginals.view(-1, num_tags).max(1)[1].long())

# Computes CE-Loss with respect to the unary_marginal probability distribution
def cross_entropy_loss(logits, unary_marginals, mask):
    empirical_probs = softmax(logits, dim=-1).view(-1, 3)
    target_probs = unary_marginals.view(-1, 3)
    cross_entropies = -(target_probs * empirical_probs.log()).sum(1)

    # Filters masked logits, and computes the average only among unmasked ones
    return torch.dot(cross_entropies, mask.view(-1)) / mask.sum()

def entropy(logits):
    empirical_probs = softmax(logits, dim=-1)
    return -(empirical_probs * empirical_probs.log()).sum(-1)

@Model.register("wiser_crf_tagger")
class WiserCrfTagger(CrfTagger):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, text_field_embedder, encoder,
                         label_namespace, feedforward, label_encoding,
                         include_start_end_transitions, constrain_crf_decoding, calculate_span_f1,
                         dropout, verbose_metrics, initializer, regularizer)

        # Gets the kwargs needs to initialize the WISER CRF. We skip some
        # configuration checks that are checked in the super constructor
        if constrain_crf_decoding:
            labels = self.vocab.get_index_to_token_vocabulary(
                self.label_namespace)
            constraints = allowed_transitions(self.label_encoding, labels)
        else:
            constraints = None

        # Replaces the CRF created by the super constructor with the WISER CRF
        self.crf = WiserConditionalRandomField(
            self.num_tags, constraints,
            include_start_end_transitions=include_start_end_transitions
        )

        self.l = 100

        self.regularization_penalty = 0
        self.detection = 0
        self.masked_fraction = 0

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
        predicted_tags = [x for x, y in best_paths]
        batch_size, seq_len, num_tags = logits.size()

        # Just get the tags and ignore the score.
        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        unary_marginals = kwargs.get('unary_marginals')
        pairwise_marginals = kwargs.get('pairwise_marginals')
        vote_mask = kwargs.get('vote_mask').long()


        if unary_marginals is not None:
            # mask = mask.float() * vote_mask.float()
            # mask = mask.float()
            # values, indices = unary_marginals.max(-1)           # computes target one hot vector from unary marginals
            # indices = indices.view(-1)
            # new_mask = mask.view(-1)
            # stacked_mask = torch.stack([new_mask] * num_tags, -1)
            #
            # loss = CrossEntropyLoss()
            # output['loss'] = loss(logits.view(-1, num_tags) * stacked_mask, new_mask.long() * indices)
            ell = self.crf.expected_log_likelihood(logits=logits,
                                                mask=mask,
                                                unary_marginals=unary_marginals,
                                                pairwise_marginals=pairwise_marginals)
            output["loss"] = -ell

        if tags is not None:
            if unary_marginals is None:
                log_likelihood = self.crf(logits, tags, mask)
                output['loss'] = -log_likelihood

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

        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output
