from typing import List, AnyStr

import pandas as pd
import pickle 
import re


WIKI_DATASET_TRAIN_RAW_PATH = "dataset//wikitext_train_raw.pickle"
WIKI_DATASET_TEST_RAW_PATH = "dataset//wikitext_test_raw.pickle"


class WikiTextParser:
    header_regex = re.compile(r" = .+ = .*")
    article_regex = re.compile(r"( = ){1}[^=;]+( = ){1}")


    def __init__(self, data: List[AnyStr]):
        data = self.__parse_data_to_dataframe(data)
        print(data)


    def __parse_data_to_dataframe(self, data: List[AnyStr]):
        article_id = -1
        header_id = 0
        line_id = 0
        index_tuples = []
        text_lines = []

        for line in data:
            if re.match(self.article_regex, line):
                article_id += 1
                header_id = 0
                line_id = 0

            elif re.match(self.header_regex, line):
                header_id += 1
                line_id = 0

            elif line != '':
                index_tuples.append((article_id, header_id, line_id))
                text_lines.append(line)
                line_id += 1

        index = pd.MultiIndex.from_tuples(index_tuples, names=(['article', 'part', 'line']))
        return pd.DataFrame(text_lines, columns=['text'], index=index)


if __name__ == '__main__':
    data: List[AnyStr] = pickle.load(open(WIKI_DATASET_TRAIN_RAW_PATH, "rb"))

    data_parser = WikiTextParser(data)