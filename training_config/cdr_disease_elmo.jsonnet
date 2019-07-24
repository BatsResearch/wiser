// Configuration for the CDR Disease NER model with ELMo, modified slightly
// from the conll configuration
{
  "random_seed": std.extVar("RANDOM_SEED"),
  "numpy_seed": std.extVar("RANDOM_SEED"),
  "pytorch_seed": std.extVar("RANDOM_SEED"),
  "dataset_reader": {
    "type": "weak_label",
    "token_indexers": {
        "tokens": {
          "type": "single_id",
        },
        "token_characters": {
          "type": "characters",
          "min_padding_length": 3
        },
        "elmo": {
          "type": "elmo_characters"
        }
    },
    "split_sentences": false
  },
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path":  std.extVar("DEV_PATH"),
  "model": {
    "type": "wiser_crf_tagger",
    "label_encoding": "IOB1",
    "dropout": 0.5,
    "include_start_end_transitions": true,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
        "elmo":{
            "type": "elmo_token_embedder",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0
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
      "input_size": 1202,
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
    "batch_size": 16
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
    "cuda_device": 0
  }
}
