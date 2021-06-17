# WISER

Welcome to WISER (*Weak and Indirect Supervision for Entity Recognition*), a system for training sequence tagging models, particularly neural networks for named entity recognition (NER) and related tasks. WISER uses *weak supervision* in the form of rules to train these models, as opposed to hand-labeled training data.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://travis-ci.com/BatsResearch/wiser.svg?branch=master)](https://travis-ci.com/BatsResearch/wiser)
[![Documentation Status](https://readthedocs.org/projects/wiser-system/badge/?version=latest)](http://wiser-system.readthedocs.io/?badge=latest)

The original WISER paper can be accessed [here](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf).

## Benchmarks

| Method | NCBI-Disease (F1) | BC5CDR (F1) | LaptopReview (F1) |
| ------------- |-------------| -----| -----|
| AutoNER | 75.52 | 82.13 | 65.44 |
| Snorkel | 73.41 | 82.24 | 63.54 |
| WISER | **79.03** | **82.94** | **69.04** |

## Getting Started

These instructions will WISER up and running on your local machine to develop your own pipelines for weakly supervised for sequence tagging tasks.

### Installing

WISER requires Python 3.7. To install the required dependencies, please run

```
pip install -r requirements.txt
```

Or alternatively

```
conda install --file requirements.txt
```

Then, inside the *wiser* directory, please run

```
pip install .
```

## Getting Started

Refer to *tutorial/introduction* for a comprehensive introduction to using WISER to train end-to-end frameworks with weak supervision. More tutorials coming soon!

Once you're comfortable with the WISER framework, we recommend looking at our [FAQ](https://github.com/BatsResearch/wiser/blob/master/FAQ.md) for strategies on how to write rules and debug your pipeline.

## Citation

Please cite the following paper if you are using our tool. Thank you!

[Esteban Safranchik](https://www.linkedin.com/in/safranchik/), Shiying Luo, [Stephen H. Bach](http://cs.brown.edu/people/sbach/). "Weakly Supervised Sequence Tagging From Noisy Rules". In 34th AAAI Conference on Artificial Intelligence, 2020.

```
@inproceedings{safranchik2020weakly,
  title = {Weakly Supervised Sequence Tagging From Noisy Rules}, 
  author = {Safranchik, Esteban and Luo, Shiying and Bach, Stephen H.}, 
  booktitle = {AAAI}, 
  year = 2020, 
}
```

