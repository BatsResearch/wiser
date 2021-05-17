// Configuration for the CONLL model from AllenAI, modified slightly
// and switched to BERT features
{
  "random_seed": 0,
  "numpy_seed": 0,
  "pytorch_seed": 0,
  "dataset_reader": {
    "type": "weak_label",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
    },
  },
  "train_data_path": "output/generative/hmm/train_data.p",         // If necessary, change to your own data path
  "validation_data_path": "output/generative/hmm/dev_data.p",
  "test_data_path": "output/generative/hmm/test_data.p",
  "evaluate_on_test": true,
  "model": {
    "type": "wiser_crf_tagger",
    "label_encoding": "IOB1",
    "dropout": 0.5,
    "include_start_end_transitions": true,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "token_characters": ["token_characters"],
      },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased"
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 128,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu"
            }
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768 + 128,
      "hidden_size": 200,
      "num_layers": 3,
      "dropout": 0.5,
      "bidirectional": true
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.1
        }
      ]
    ],
    "use_tags": false // Only change to true if tags exists and you're planning to run the fully-supervised model
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 50,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": -1 // Change to 0 if you have a CUDA device
  }
}
