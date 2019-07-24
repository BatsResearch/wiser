import argparse
from labelmodels import NaiveBayes, HMM, LinkedHMM, LearningConfig
import pickle
from util import train_generative_model

parser = argparse.ArgumentParser(description="Trains a generative model.")
parser.add_argument('train_data', type=str,
                    help='path to pickled training data')
parser.add_argument('dev_data', type=str,
                    help='path to pickled development data')
parser.add_argument('model_type', type=str,
                    choices=("nb", "hmm", "link_hmm"),
                    help='type of generative model to train')
parser.add_argument('acc_prior', type=float,
                    help='model accuracy prior')
parser.add_argument('balance_prior', type=float,
                    help='model balance prior')
parser.add_argument('random_seed', type=int,
                    help='seed to use for shuffling training data')

args = parser.parse_args()

with open(args.train_data, 'rb') as f:
    train_data = pickle.load(f)

with open(args.dev_data, 'rb') as f:
    dev_data = pickle.load(f)

num_classes = 2
init_acc = 0.8
epochs = 10
label_to_ix = {'ABS': 0, 'I': 1, 'O': 2}

labeling_functions = set()
linking_functions = set()
for doc in train_data:
    for name in doc['WISER_LABELS'].keys():
        labeling_functions.add(name)
    for name in doc['WISER_LINKS'].keys():
        linking_functions.add(name)

if args.model_type == 'nb':
    model = NaiveBayes(num_classes,
                       len(labeling_functions),
                       init_acc=init_acc,
                       acc_prior=args.acc_prior,
                       balance_prior=args.balance_prior)
elif args.model_type == 'hmm':
    model = HMM(num_classes,
                       len(labeling_functions),
                       init_acc=init_acc,
                       acc_prior=args.acc_prior,
                       balance_prior=args.balance_prior)
elif args.model_type == 'link_hmm':
    model = LinkedHMM(num_classes,
                      len(labeling_functions),
                      len(linking_functions),
                      init_acc=init_acc,
                      acc_prior=args.acc_prior,
                      balance_prior=args.balance_prior)
else:
    raise ValueError(args.model)

config = LearningConfig()
p, r, f1 = train_generative_model(model, train_data, dev_data,
                                  epochs, label_to_ix, config)

print(args)
print(p)
print(r)
print(f1)
