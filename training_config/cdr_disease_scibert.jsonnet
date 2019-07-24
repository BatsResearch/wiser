// Configuration adapted from https://gist.github.com/joelgrus/7cdb8fb2d81483a8d9ca121d9c617514
// for BERT embeddings CRF tagger
// Changed settings to use those specified in EBM-NLP paper
{
  "random_seed": std.extVar("RANDOM_SEED"),
  "pytorch_seed": std.extVar("RANDOM_SEED"),
  "numpy_seed": std.extVar("RANDOM_SEED"),
  "dataset_reader": {
//    "type": "weak_label",
    "type": "ebmnlp",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": std.extVar("BERT_VOCAB"),
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("DEV_PATH"),
  "test_data_path": std.extVar("DEV_PATH"),
//"evaluate_on_test": true,
  "model": {
    "type": "wiser_crf_tagger",
    "label_encoding": "IOB1",
    "dropout": 0.5,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
            "token_characters": ["token_characters"]
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": std.extVar("BERT_WEIGHTS")
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
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 15,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0
  }
}
