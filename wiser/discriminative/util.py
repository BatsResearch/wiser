import os
import random


def train_discriminative_model(train_data_path, dev_data_path, test_data_path,
                               training_config_path, path_to_save='./discriminative_output',
                               cuda_device=-1, use_tags=False, seed=None):

    if seed is None:
        seed = random.randint(0, 10e6)

    os.environ['RANDOM_SEED'] = str(seed)
    os.environ['TRAIN_PATH'] = str(train_data_path)
    os.environ['DEV_PATH'] = str(dev_data_path)
    os.environ['TEST_PATH'] = str(test_data_path)
    os.environ['CUDA_DEVICE'] = str(cuda_device)
    os.environ['USE_TAGS'] = str(use_tags)


    os.system('allennlp train %s -f -s %s --include-package wiser' %
              (training_config_path, path_to_save))
