# Part 2: Generative Models
In this part of the tutorial, we will take the results of the labeling functions from part 1 and learn a generative model that combines them.

We will start by reloading the data with the labeling function outputs from part 1.

## Reloading Data
```python
import pickle

with open('output/tmp/train_data.p', 'rb') as f:
    train_data = pickle.load(f)

with open('output/tmp/dev_data.p', 'rb') as f:
    dev_data = pickle.load(f)
    
with open('output/tmp/test_data.p', 'rb') as f:
    test_data = pickle.load(f)
```

## Reinspecting Data
We can now browse the data with all of the tagging rule annotations. Browse the different tagging rules and their votes on the dev data.
```python
from wiser.viewer import Viewer
Viewer(dev_data, height=120)
```
We can inspect the raw precision, recall, and F1 score using an unweighted combination of tagging rules with score_labels_majority_vote.
```python
from wiser.eval import score_labels_majority_vote
score_labels_majority_vote(dev_data)
```

## Generative Model
To weight the tagging and linking rules according to estimated accuracies, need to train a generative model.

### Defining a Generative Model
We now need to declare a generative model. In this tutorial, we will be using the *linked HMM*, a model that makes use of linking rules to model dependencies between adjacent tokens. You may find other generative models in `labelmodels`.

Generative models have the following hyperparameters:

- Initial Accuracy (init_acc) is the initial estimated tagging and link-ing rule accuracy, also used as the mean of the prior distribution of the model parameters.
- Strength of Regularization (acc_prior) is the weight of the regularizer pulling tagging and linking rule accuracies toward their initial values.
- Balance Prior (balance_prior) is used to regularize the class prior in Naive Bayes or the initial class distribution for HMM and Linked HMM, as well as the transition matrix in those methods, towards a more uniform distribution.

We generally recommend running a grid search on the generative model hyperparameters to obtain the best performance. For more details on generative models and the *linked HMM*, please refer to our paper.

```python
from labelmodels import LinkedHMM
from wiser.generative import Model

model = Model(LinkedHMM, init_acc=0.95, acc_prior=50, balance_prior=100)
```

### Training a Generative Model

Once we're done creating our generative model, we're ready to begin training! We first need to create a `LearningConfig` to specify the training configuration for the model.

```python
from labelmodels import LearningConfig

config = LearningConfig()
config.epochs = 5
```

Then, we must pass the config object to the `train` , alongside the training and development data.

```python
# Outputs the best development score
model.train(config, train_data=train_data, dev_data=dev_data)
```

### Evaluating a Generative Model
We can easily evaluate the performance of any generative model using the function `evaluate` function. Here, we'll evaluate our *linked HMM* on the test set.

```python
model.evaluate(test_data)
```
If you've been following this tutorial, test precision should be around 75.6%, and test F1 should be around 64%.



### Saving the Output of the Generative Model

After implementing your generative model, you need to save its probabilistic training labels. The `save_probabilistic_output` wrapper function will save the probabilistic tags to the specified directory. We will later use these labels in the next part of the tutorial to train a recurrent neural network.


```python
model.save_output(data=train_data, path='output/generative/link_hmm/train_data.p', save_distribution=True)
model.save_output(data=dev_data, path='output/generative/link_hmm/dev_data.p', save_distribution=True, save_tags=True)
model.save_output(data=test_data, path='output/generative/link_hmm/test_data.p', save_distribution=True, save_tags=True)
```

