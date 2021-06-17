# WISER

WISER (Weak and Indirect Supervision for Entity Recognition) is a library that trains sequence-to-sequence models using programmatic weak supervision. Users provide rules / heuristics (in the form of labeling functions) to train neural networks for sequence labeling tasks such as named entity recognition. 

## Installation

WISER requires Python 3.7. To use WISER, please run 

```bash
git clone https://github.com/BatsResearch/wiser.git
cd wiser
pip3 install -r requirements.txt
```

If you are using conda environment, you can alternatively run 
```bash
git clone https://github.com/BatsResearch/wiser.git
cd wiser
conda install --file requirements.txt
```

## Debugging

- If you run into the error of `TypeError: Params.pop: key must be a supertype of <class 'inspect._empty'> but is <class 'str'>`, you can fix the issue by running `pip3 install overrides==4.1.2`.



