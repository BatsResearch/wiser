# WISER

Welcome to WISER (*Weak and Indirect Supervision for Entity Recognition*), a system for training sequence-to-sequence models, particularly neural networks for named entity recognition (NER) and related tasks. WISER uses *weak supervision* in the form of rules to train these models, as opposed to hand-labeled training data.

## Getting Started

These instructions will WISER up and running on your local machine to develop your own pipelines for weakly supervised for sequence tagging tasks.

### Prerequisites

To get the code for generative models and other rule-aggregation methods, please download and install the latest version of [labelmodels](https://https://github.com/BatsResearch/labelmodels), our lightweight implementation of generative label models for weakly supervised machine learning.

### Installing

In your virtual enviornment, please run the *labelmodels/setup.py* and *wiser/setup.py* scripts to install the corresponding dependencies.

```
python3 setup.py install
```

Then, to install other dependencies required to run WISER, run

```
pip3 install -r requirements.txt
```

## Authors

* **Esteban Safranchik**
* **Shiying Luo**
* **Stephen H. Bach**
