# Part 3: Neural Networks

In this part of the tutorial, we use the probabilistic labels of the generative model to train a state-of-the-art recurrent neural network. We will use a bi-LSTM-CRF to improve the generalization of the linked HMM model.

## Training a Neural Network
To train your bi-LSTM-CRF network, you first need to define an [AllenNLP configuration file](https://gist.github.com/joelgrus/7cdb8fb2d81483a8d9ca121d9c617514). These files provide a flexible way of defining neural networks, detailing all the model parameters. For this tutorial, we have already provided a configuration file in [training_config/tutorial.jsonnet](https://github.com/BatsResearch/wiser/blob/master/tutorials/introduction/training_config/tutorial.jsonnet), which specifies the data path, LSTM encoder configuration, and cuda device, among other parameters. Feel free to take a look at it and edit it for your own experiments.

You may also define other neural networks to learn from the probabilistic output of the generative model. Our implementation uses a bi-LSTM-CRF, but you can build many interesting pipelines using different models as long as they optimize a noise-aware loss function (see `modules/conditional_random_field`).

To begin training, we need to call AllenNLP's `train_model_from_file` method and pass it in the location of the configuration file and the output directory.

```python
from wiser.data.dataset_readers import weak_label   # You need to import weak_label and WiserCrfTagger
from wiser.models import WiserCrfTagger             # since they are used in the training config. file
from allennlp.commands.train import train_model_from_file

train_model_from_file(parameter_filename='training_config/tutorial.jsonnet',
                      serialization_dir='output/discriminative/link_hmm', 
                      file_friendly_logging=True, force=True)
```
Once you finish training your discriminative model, you will find it's output scores in the `serialization_dir`.

Then, open the `metrics.json` file. If you've followed this tutorial closely, the test F1 should be around 70%. That's more than a six point increase with respect to the generative model!

Recall should have increased the most, suggesting an improvement in the generalization of the pipeline.


## Predicting Tags

To use your WISER model to make predictions, you can call AllenNLP's Predictor class. You will need to pass in the path to the `model.tar.gz` file inside your output directory, alongside the `sentence-tagger` predictor configuration.

```python
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

predictor = Predictor.from_path(archive_path='output/discriminative/link_hmm/model.tar.gz', 
                                predictor_name='sentence-tagger')
```

As a small demonstration of our pipeline, we will use SpaCy's word splitter to show individual tag predictions of the discriminative model.

```python
tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=False)

sentence = 'The movie The Lord of the Rings: The Return of the King (2003) \
            won all 11 awards for which it was nominated, \
            including the Emmy Award for Best Picture'

# Prints all tokens in the sentence, alongside their predicted tags
for match in zip(tokenizer.split_words(sentence), predictor.predict(sentence)['tags']):
    print(match)
```

Alternatively, you can use the AllenNLP `predict` command to predict entire .json files at once. You will need to use the `--predictor sentence-tagger and --include-package wiser` arguments if you use this command.

Congratulations! You are done with the introductory tutorial!















