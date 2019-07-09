import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from .lf import LinkingFunction


class ElmoLinkingFunction(LinkingFunction):
    def __init__(self, threshold):
        super().__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        self.threshold = threshold

    def apply_instance(self, instance):
        sentences = [[x.text for x in instance['tokens']]]
        character_ids = batch_to_ids(sentences)
        emb = self.elmo(character_ids)

        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            num = torch.dot(emb['elmo_representations'][0][0, i - 1, :],
                            emb['elmo_representations'][0][0, i, :])
            denom = torch.norm(emb['elmo_representations'][0][0, i - 1, :])
            denom *= torch.norm(emb['elmo_representations'][0][0, i, :])
            sim = num / denom
            if sim > self.threshold:
                links[i] = 1

        return links

    def _get_lf_name(self):
        return "ElmoLinkingFunction-" + str(self.threshold)
