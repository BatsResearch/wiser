// Configuration for the CONLL model from AllenAI, modified slightly
// and switched to BERT features
{
  "random_seed": std.extVar("RANDOM_SEED"),
  "numpy_seed": std.extVar("RANDOM_SEED"),
  "pytorch_seed": std.extVar("RANDOM_SEED"),
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
//    "split_sentences": false
  },
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path":  std.extVar("DEV_PATH"),
  "test_data_path": std.extVar("TEST_PATH"),
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
      "num_layers": 2,
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
    ]
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
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": std.extVar("CUDA_DEVICE"),
  }
}
