# WISER

Welcome to WISER (*Weak and Indirect Supervision for Entity Recognition*), a system for training sequence-to-sequence models, particularly neural networks for named entity recognition (NER) and related tasks. WISER uses *weak supervision* in the form of rules to train these models, as opposed to hand-labeled training data.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

In your virtual environment, please install the required dependencies using

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

Refer to *tutorial/introduction* for a comprehensive introduction to using WISER to train end-to-end frameworks with weak supervision. 

More tutorials coming soon!

## Citation

Please cite the following paper if you are using our tool. Thank you!

[Esteban Safranchik](https://twitter.com/safranchik_e), Shiying Luo, [Stephen H. Bach](https://twitter.com/stevebach). "Weakly Supervised Sequence Tagging From Noisy Rules". In 34th AAAI Conference on Artificial Intelligence, 2020.

```
@inproceedings{safranchik2020weakly,
  title = {Weakly Supervised Sequence Tagging From Noisy Rules}, 
  author = {Safranchik, Esteban and Luo, Shiying and Bach, Stephen H.}, 
  booktitle = {AAAI}, 
  year = 2020, 
}
```
