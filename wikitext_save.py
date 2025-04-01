from typing import List, AnyStr
import datasets
import pickle


WIKI_DATASET_TRAIN_RAW_PATH = "dataset//wikitext_train_raw.pickle"
WIKI_DATASET_TEST_RAW_PATH = "dataset//wikitext_test_raw.pickle"


if __name__ == '__main__':

    data = datasets.load_dataset(path='wikitext', name='wikitext-103-raw-v1', split='train')['text']
    pickle.dump(data, open(WIKI_DATASET_TRAIN_RAW_PATH, 'wb'))

    data = datasets.load_dataset(path='wikitext', name='wikitext-103-raw-v1', split='test')['text']
    pickle.dump(data, open(WIKI_DATASET_TEST_RAW_PATH, 'wb'))

