# Part 1: Tagging and Linking Rules
WISER is an add-on to [AllenNLP](https://github.com/allenai/allennlp), a great framework for natural language processing. That means we can use their tools for working with data.

## Loading Data
Let's start by loading the Media dataset, a new dataset we created just for this tutorial.

```python
from wiser.data.dataset_readers import MediaDatasetReader
import os

dataset_reader = MediaDatasetReader()
train_data = dataset_reader.read('data/wikipedia/unlabeled_train.csv')
dev_data = dataset_reader.read('data/wikipedia/labeled_dev.csv')
test_data = dataset_reader.read('data/wikipedia/labeled_test.csv')

# Here, we just set how many documents we'll process for automatic testing- you can safely ignore this!
if 'CI' in os.environ:  
    train_data = train_data[:10]
    dev_data = dev_data[:10]
    test_data = test_data[:10]
else:
    train_data = train_data[:750]

# We must merge all partitions to apply the rules
data = train_data + dev_data + test_data
```

The easiest way to use WISER with other data sets is to implement a new subclass of AllenNLP's [DatasetReader](http://docs.allennlp.org/v0.9.0/api/allennlp.data.dataset_readers.dataset_reader.html). We have some additional examples in the package `wiser.data.dataset_readers`.


## Inspecting Data
Once the data is loaded, we can use a WISER class called `Viewer` to inspect the sentences and tags.
```python
from wiser.viewer import Viewer
Viewer(dev_data, height=120)
```

You can use the left and right buttons to flip through the items in dev_data, each of which is an AllenNLP [Instance](http://docs.allennlp.org/v0.9.0/api/allennlp.data.instance.html?highlight=instance#module-allennlp.data.instance). The highlighted spans are the entities, and you can hover over each one with your cursor to see whether it is an award (**AWD**), or a media performance **PERF**.

The drop-down menu selects which source of labels is displayed. Currently only the gold labels from the benchmark are available, but we will add more soon.

Advance to the instance at index 1 to see an example with multiple entities of different classes. You can access the underlying tags too by hovering over particular tokens.

Notice that WISER uses the [IOB1 tagging scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)), meaning that entities are represented as consecutive tags beginning with I. Many data sets use subsequent characters for different classes, for example **-AWD** for awards and **-PERF** for movies, T.V. shows, or theatre plays. The **O**, or other tag, means that the token is not part of an entity. There is also a special set of tags beginning with **B** (like those beginning with **I**) that are used to start a new entity that immediately follows another of the same class without an **O** tag in between.

## Tagging Rules
Tagging rules are functions that map unlabeled text instances to sequences of labels. We can define our own tagging rules by writing small functions that look at sequences of instance tokens, and vote on their correponding tags. Let's first import the `TaggingRule` class from `wiser.rules`.

```python
from wiser.rules import TaggingRule
```

### Writing Simple Tagging Rules
From inspecting instance 11, we know tokens proper nouns followed by a year between parentheses are likely tagged as movies. For instance, the token `Friends` in the span `Friends (1994 - 2004)` should be tagged **I-PERF**. Let's write our first tagging rule to reflect this!

```python
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
```

We can also write a tagging rule to identify award categories like for *Oustanding Lead Actress* in award spans such as *BAFTA Award for Oustanding Lead Actress*. Categories are generally preceded by capitalized letters, and follow with the strings *for Oustanding* or *for Best*. Please refer instance at index 1 for an example of this.

```python
class AwardCategory(TaggingRule):
    
    def apply_instance(self, instance):

        tokens = [t.text for t in instance['tokens']]
        labels = ['ABS'] * len(tokens)
        
        for i in range(len(tokens)-2):
            if tokens[i].istitle() and tokens[i+1] == 'for' and tokens[i+2] in {'Best', 'Oustanding'}:
                labels[i+1] = 'I-AWD'
                labels[i+2] = 'I-AWD'
               
        return labels

tr = AwardCategory()
tr.apply(data)
```

### Tagging Function Helpers
You can also use existing tagging functions and helpers available at `wiser.rules`. The `DictionaryMatcher` is a tagging function helper that allows us to quickly create a new rule that votes on any element found in a set of characters or words.

```python
from wiser.rules import DictionaryMatcher
```

Any token spelling `Award` , `Awards`, `Prize` or `Cup` should be tagged as an award. Be mindful of capitalization, since awards are proper nouns!

```python
award_keywords = [['Award'], ['Awards'], ['Prize'], ['Cup']]
                  
tr = DictionaryMatcher("AwardKeywords", terms=award_keywords, i_label="I-AWD", uncased=False)
tr.apply(data)
```

A good trick to developing efficient sequence taggers is to also generate negative supervision in the form of **O** tags. We can write the first function of this kind to tag some punctuations signs!

```python
non_entity_punctuation_chars = {'.', ';', '(', ')'}

tr = DictionaryMatcher("Non-EntityPunctuation", terms=non_entity_punctuation_chars, i_label="O")
tr.apply(data)
```

We recommend going over the data and identifying a few false positive tokens. That is, tokens that are similar to entities but are not (e.g., capitalized tokens such as studio names). We will also write a `DictionaryMatcher` identify some common false positives and tag them as such:

```python
common_false_positives = [['network'], ['netflix'], ['hulu'], ['bbc'], ['fox'], 
                          ['disney'], ['hbo'], ['CBS'], ['channel'], ['american'], 
                         ['showtime'], ['productions'], ['TV']]

tr = DictionaryMatcher("CommonFalsePositives", terms=common_false_positives, i_label="O", uncased=True)
tr.apply(data)
```

### Looking at Previous Tagging Rules
You can also develop more complex tagging rules by looking at previous tagging rule votes using the `WISER_LABELS` field. However, be mindful of the order in which you run the tagging functions.

In the following example, we will write a tagging rule to identify performances based on adjacent tags such as `series` or `show` (e.g., *The TV series The Mandalorian* or *Kung-Fu Panda franchise*). However, we also want to avoid tagging common false positives such as TV), which is why we will reference the output votes of the `CommonFalsePositives` rule.

```python
movie_keywords = {'trilogy', 'saga', 'series', 'miniseries', 
                'show', 'opera', 'drama', 'musical', 'sequel',
                'prequel', 'franchise', 'thriller', 'sitcom'}

class MovieKeywords(TaggingRule):
    
    def apply_instance(self, instance):

        tokens = [t.text for t in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        # List of tag votes of CommonFalsePositives rule
        false_positives = [t for t in instance['WISER_LABELS']['CommonFalsePositives']]
        
        for i in range(len(tokens)):
            if tokens[i].lower() in movie_keywords:
                """
                    We will only tag a word as a movie if 
                    the CommonFalsePositives asbtained from voting it.
                    
                    We also want to avoid award names 
                    like "... Musical Drama", etc.
                """ 
                
                # Keywords followed by movies (e.g., Kung-Fu Panda franchise)
                if i < len(tokens) and tokens[i+1].istitle() and false_positives[i+1] == 'ABS':
                    if tokens[i+1].lower() not in movie_keywords:
                        labels[i+1] = 'I-PERF'
                       
                # Movies followed by keywords 
                elif i > 0 and tokens[i-1].istitle() and false_positives[i-1] == 'ABS':
                    if tokens[i-1].lower() not in movie_keywords:
                        labels[i-1] = 'I-PERF'
        return labels

tr = MovieKeywords()
tr.apply(data)
```

### Using Existing Models

WISER also allows us to implement existing machine learning systems to provide more accurate weak supervision. You can import existing classifiers or NLP tools to improve your current model.

For our next tagging rule, we will use spaCy's [part-of-speech tagger](https://spacy.io/usage/linguistic-features#pos-tagging). This pre-trained model identifies sentence part-of-speech tags, and will be useful to identify many non-entity words such as lowercased nouns, verbs, and adjectives.


```python
import spacy

nlp = spacy.load("en_core_web_sm")
tagger = nlp.create_pipe("tagger")

non_entity_lowercases = {'NOUN', 'VERB', 'ADJ', 'SPACE', 'NUM'}

class NonEntityWords(TaggingRule):
    
    def apply_instance(self, instance):

        tokens = [t.text for t in instance['tokens']]
        
        # We obtain the parts-of-speech from SpaCy
        parts_of_speech = [token[0].pos_ for token in nlp.pipe(tokens)]        
        labels = ['ABS'] * len(tokens)

        for i, (token, pos) in enumerate(zip(tokens, parts_of_speech)):
            if pos in non_entity_lowercases and not token.istitle():
                labels[i] = 'O'
                
        return labels
#This rule takes a while to run, so we skip applying it for our testing. You can safely ignore this line.
if not 'CI' in os.environ:
    tr = NonEntityWords()
    tr.apply(data)
```


### Evaluating Tagging Rules
We can inspect the performance of individual tagging_rules by using the score_tagging_rules method.

- True positives (TP) represent the number of items correctly labeled spans belonging to a positive class (e.g. I-PERF).
- False positives (FP) are the number of items incorrectly labeled spans belonging to a positive class.
- False Negatives (FN) are the items which were not labeled as belonging to the positive class but should have been.
- Token Accuracy (Token Acc.) represents the fraction of issued votes that correctly identified a token in a positive class.
- Token Votes is the total number of times the tagging rules issued a vote belonging to a positive class.

```python
from wiser.eval import score_tagging_rules
score_tagging_rules(dev_data)
```

A good rule of thumb is to write tagging rules whose accuracy is above 90%.


## Linking Rules
Linking rules are simple functions that vote on whether two or more adjacent tokens belong should belong to the same entity. To get started with linking rules, you can import the `LinkingRule` class from `wiser.rules`

```python
from wiser.rules import LinkingRule
```


### Writing Linking Rules
Tagging rules do not always correctly vote on all the tokens in multi-span entities. For instance, the `MovieYear` tagging rule only tags the last token in a movie span. Given the text span *The Great Gatsby (2013)*, it only identifies the token `Gatsby` as **I-PERF**.

Our job is to ensure that the entire class spans are tagged correctly. Therefore, we can start by writing a linking rule to indicate that consecutively capitalized words should share the same tag. Therefore, voting that `The` and `Great` share the same tag as `Gatsby` would tag the entire movie name as **I-PERF**, rather than the last token.

```python
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
```

In our data we have also observed several movie and award names that have hyphens, colons or semicolons (e.g. *Avengers: Endgame*). We can write a linking rule to indicate that these linking punctuation characters, along with their preceding and succeeding tokens, should all be a part of the same entity.

```python
linkers = {':', ';', '-'}

class PunctuationLinkers(LinkingRule):

    def apply_instance(self, instance):
        tokens = [t.text for t in instance['tokens']]
        links = [0] * len(tokens)
        
        for i in range(1, len(tokens)-1):
            if tokens[i] in linkers:
                
                # The linking punctuation character and it's succeeding character
                # share the same tag as the preceding one at index "i-1"
                links[i] = 1
                links[i+1] = 1
        return links

lr = PunctuationLinkers()
lr.apply(data)
```

Similarly, we can write a rule to indicate that contractions share the same tag with the token preceding them.

```python
contraction_suffixes = {'\'s', '\'nt', '\'ve', '\'', '\'d'}

class Contractions(LinkingRule):

    def apply_instance(self, instance):
        tokens = [t.text for t in instance['tokens']]
        links = [0] * len(tokens)
        
        for i in range(1, len(tokens)):
            if tokens[i] in contraction_suffixes:
                links[i] = 1
        return links

lr = Contractions()
lr.apply(data)
```

We can also link noun phrases that using a list of common prepositions in award and movie names. These prepositions are part of award and movie names, and are usually lowercase and adjacent to or other prepositions or capitalized words. For example, *Golden Globe for Best Actor* or *Guardians of the Galaxy*.

```python
common_prepositions = {'a', 'the', 'at', 'with', 'of', 'by', '&', 'with'}

class CommonPrepositions(LinkingRule):

    def apply_instance(self, instance):
        tokens = [t.text for t in instance['tokens']]
        links = [0] * len(tokens)
        
        for i in range(1, len(tokens)-1):
            if tokens[i] in common_prepositions:
                if tokens[i-1].istitle() or tokens[i-1] in common_prepositions:
                    if tokens[i+1].istitle() or tokens[i+1] in common_prepositions:
                        links[i] = 1
                        links[i+1] = 1
        return links

lr = CommonPrepositions()
lr.apply(data)
```

### Linking Rule Helpers
Similar to tagging rules, we have linking rule helpers available at `wiser.rules`. For the next linking rule, we will use the `ElmoLinkingRule`, a rule that vectorizes tokens using [Elmo](https://allennlp.org/elmo) and links those with a cosine similaritiy larger than a given threshold.

```python
from wiser.rules import ElmoLinkingRule

lr = ElmoLinkingRule(0.8)
#This rule takes a while to run, so we skip applying it for our testing. You can safely ignore this line.
if not 'CI' in os.environ:
    lr.apply(data)
```

### Evaluating Linking Rules

Similar to tagging rules, we can evaluate the accuracy of our linking rules using the score_linking_functions method.
- Entity Links represents the number of correct links generated for positive classes.
- Non-Entity Links represents the number of correct links generated for negative classes (e.g., O tags).
- Incorrect links represent the total number of incorrectly generated links.
- Accuracy represents the fraction of issued links that identified correct links.

```python
from wiser.eval import score_linking_rules
score_linking_rules(dev_data)
```

Once more, a good rule of thumb is to have all linking rules with an accuracy above 90%.

## Removing Tagging and Linking Rules
When developing sequence tagging pipelines, you will often need to remove tagging or linking rules you've previously applied to the data. The `rules.remove_rule` method deletes all occurrences of a tagging or linking rule vote in a given dataset.

```python
from wiser.rules import remove_rule
```

To show how the `remove_rule` method works, let's first create a dummy tagging rule.
```python
class DummyRule(TaggingRule):
    
    def apply_instance(self, instance):
        tokens = [t.text for t in instance['tokens']]
        return ['ABS'] * len(tokens)
        
tr = DummyRule()
tr.apply(data)
```

We can observe that this rule has been applied to the development set.

```python
score_tagging_rules(dev_data)
```

Passing the rule name to the `remove_rule` method will eliminate it from the given dataset. This step can also be applied to linking rules.

```python
remove_rule(data, 'DummyRule') # Don't forget to pass the entire dataset
score_tagging_rules(dev_data)
```

## Saving Progress

We can use `pickle` to store the data with the tagging and linking rules applied to it

```python
import pickle

with open('output/tmp/train_data.p', 'wb') as f:
    pickle.dump(train_data, f)

with open('output/tmp/dev_data.p', 'wb') as f:
    pickle.dump(dev_data, f)

with open('output/tmp/test_data.p', 'wb') as f:
    pickle.dump(test_data, f)
```

You have completed part 1! Now you can move on to part 2.
