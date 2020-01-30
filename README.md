# WISER

Welcome to WISER (*Weak and Indirect Supervision for Entity Recognition*), a system for training sequence-to-sequence models, particularly neural networks for named entity recognition (NER) and related tasks. WISER uses *weak supervision* in the form of rules to train these models, as opposed to hand-labeled training data.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Benchmarks

| Method | NCBI-Disease (F1) | BC5CDR (F1) | LaptopReview (F1) |
| ------------- |-------------| -----| -----|
| AutoNER | 75.52 | 82.13 | 65.44 |
| Snorkel | 73.41 | 82.24 | 63.54 |
| WISER | **79.03** | **82.94** | **69.04** |

## Getting Started

These instructions will WISER up and running on your local machine to develop your own pipelines for weakly supervised for sequence tagging tasks.

### Prerequisites

To get the code for generative models and other rule-aggregation methods, please download and install the latest version of [labelmodels](https://https://github.com/BatsResearch/labelmodels), our lightweight implementation of generative label models for weakly supervised machine learning.

### Installing

In your virtual enviornment, go to the *labelmodels* and run

```
pip install .
```

Do the same thing on the *wiser* repo, and then install the remaining dependencies with

```
pip install -r requirements.txt
```

Then, download the small [spaCy English dictionary](https://spacy.io/models/en) using

```
python3 -m spacy download en_core_web_sm
```

## Getting Started

Please *tutorial/introduction* for a comprehensive introduction to using WISER to train end-to-end frameworks with weak supervision. 

More tutorials coming soon!

## Citation

Please cite the following paper if you are using our tool. Thank you!

Safranchik Esteban, Shiying Luo, Stephen H. Bach. "Weakly Supervised Sequence Tagging From Noisy Rules". In 34th AAAI Conference on Artificial Intelligence, 2020.

```
@inproceedings{safranchik2020weakly,
  title = {Weakly Supervised Sequence Tagging From Noisy Rules}, 
  author = {Safranchik, Esteban and Luo, Shiying and Bach, Stephen H.}, 
  booktitle = {AAAI}, 
  year = 2020, 
}
```
