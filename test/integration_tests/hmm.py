from wiser.data.dataset_readers import MediaDatasetReader
from wiser.viewer import Viewer

dataset_reader = MediaDatasetReader()
train_data = dataset_reader.read('data/wikipedia/unlabeled_train.csv')
dev_data = dataset_reader.read('data/wikipedia/labeled_dev.csv')
test_data = dataset_reader.read('data/wikipedia/labeled_test.csv')

# In this tutorial we will use only 750 instances of the training data
train_data = train_data[:10]
dev_data = dev_data[:10]
test_data = test_data[:10]

# We must merge all partitions to apply the rules
data = train_data + dev_data + test_data

from wiser.rules import TaggingRule

class MovieYear(TaggingRule):
    
    def apply_instance(self, instance):

        # Creates a list of tokens
        tokens = [t.text for t in instance['tokens']]
        
        # Initializes a list of ABS (abstain) label votes 
        labels = ['ABS'] * len(tokens)
        
        for i in range(len(tokens)-2):    
            # Tags proper nouns followed by a number between parentheses
            if tokens[i].istitle() and tokens[i+1] == '(' and tokens[i+2].isdigit():
                labels[i] = 'I-PERF'
               
        # Returns the modified label vote list
        return labels

# Applies the tagging rule to all dataset instances 
tr = MovieYear()
tr.apply(data)

from wiser.eval.util import tagging_rule_errors
Mistakes = tagging_rule_errors(dev_data, 'MovieYear', error_type = 'fp', mode = 'span')
Viewer(Mistakes, height=120)

from wiser.eval import score_tagging_rules
score_tagging_rules(dev_data)

from wiser.rules import LinkingRule
class ConsecutiveCapitals(LinkingRule):
    
    def apply_instance(self, instance):
        tokens = [t.text for t in instance['tokens']]
        links = [0] * len(tokens)
        
        for i in range(1, len(tokens)):
            if tokens[i].istitle() and tokens[i-1].istitle():
                links[i] = 1 # token at index "i" shares tag with token at index "i-1"
        return links

lr = ConsecutiveCapitals()
lr.apply(data)

from wiser.eval import score_linking_rules
score_linking_rules(dev_data)

from wiser.rules import remove_rule
class DummyRule(TaggingRule):
    
    def apply_instance(self, instance):
        tokens = [t.text for t in instance['tokens']]
        return ['ABS'] * len(tokens)
        
tr = DummyRule()
tr.apply(data)

remove_rule(data, 'DummyRule') # Don't forget to pass the entire dataset
score_tagging_rules(dev_data)

from wiser.eval import score_labels_majority_vote
score_labels_majority_vote(dev_data)

from labelmodels import HMM
from wiser.generative import Model

model = Model(HMM, init_acc=0.95, acc_prior=50, balance_prior=100)

from labelmodels import LearningConfig

config = LearningConfig()
config.epochs = 5

# Outputs the best development score
model.train(config, train_data=train_data, dev_data=dev_data)

model.evaluate(test_data)

model.save_output(data=train_data, path='output/generative/hmm/train_data.p', save_distribution=True)
model.save_output(data=dev_data, path='output/generative/hmm/dev_data.p', save_distribution=True, save_tags=True)
model.save_output(data=test_data, path='output/generative/hmm/test_data.p', save_distribution=True, save_tags=True)

from wiser.data.dataset_readers import weak_label   # You need to import weak_label and WiserCrfTagger
from wiser.models import WiserCrfTagger             # since they are used in the training config. file
from allennlp.commands.train import train_model_from_file

train_model_from_file(parameter_filename='../../test/integration_tests/IT2.jsonnet',
                      serialization_dir='output/discriminative/hmm', 
                      file_friendly_logging=True, force=True)

from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

predictor = Predictor.from_path(archive_path='output/discriminative/hmm/model.tar.gz', 
                                predictor_name='sentence-tagger')

tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=False)

sentence = 'The movie The Lord of the Rings: The Return of the King (2003) \
            won all 11 awards for which it was nominated, \
            including the Emmy Award for Best Picture'

# Prints all tokens in the sentence, alongside their predicted tags
for match in zip(tokenizer.split_words(sentence), predictor.predict(sentence)['tags']):
    print(match)