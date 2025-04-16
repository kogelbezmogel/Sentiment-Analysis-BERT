from typing import List, AnyStr

import pandas as pd
import pickle 
import re
import emoji
import random
import math

WIKI_DATASET_TRAIN_RAW_PATH = "dataset//wikitext_train_raw.pickle"
WIKI_DATASET_TEST_RAW_PATH = "dataset//wikitext_test_raw.pickle"


class WikiTextParser:
    header_regex = re.compile(r" = .+ = .*")
    article_regex = re.compile(r"( = ){1}[^=;]+( = ){1}")
    html_tag_regex = re.compile(r'<.*?>')
    url_regex = re.compile(r'(http : / / )?www\.\S+( / \S+)*( \? \S+ = \S+( \& \S+ = \S+)*)?')
    number_regex = re.compile(r'(\d+( @[\.,]@ \d+)+)|(\d+([\.,]\d+)*)')


    def __init__(self, data: List[AnyStr] = []):
        if data:
            self.data = self.__parse_data_to_dataframe(data)

    @classmethod
    def from_parsed_data(cls, path):
        parser =  WikiTextParser()
        parser.data = pickle.load( open(path, 'rb') )
        return parser

    def to_lowercase(self):
        for index, row in self.data.iterrows():
            row['text'] = row['text'].lower()


    def numbers_to_tokens(self, token = '[NUM]'):
        for index, row in self.data.iterrows():
            row['text'] = re.sub(self.number_regex, token, row['text'])


    def urls_to_tokens(self, token = '[URL]'):
        for index, row in self.data.iterrows():
            row['text'] = re.sub(self.url_regex, token, row['text'])


    def extend_abrevations(self, abrevation_dict = None):
        pass
    

    def remove_html_tags(self):
        for index, row in self.data.iterrows():
            row['text'] = re.sub(self.html_tag_regex, '', row['text'])


    def remove_empty_lines(self):
        pass


    def remove_stopwords(stopwords_set = None):
        pass


    def save_parsed_data(self, path: AnyStr):
        pickle.dump(self.data, open(path, 'wb'))


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


def text_analysis(corpus):
    detected_nonascii = set()
    detected_emojis = set()
    detected_numbers = set()
    nonascii_counter = 0
    html_counter = 0
    url_counter = 0
    emojis_counter = 0
    numbers_counter = 0

    for index, row in corpus.iterrows():
        nonascii_found = False
        url_found = False
        html_found = False
        emojis_found = False
        numbers_found = False

        # non-ascii and emojis
        for c in row['text']:

            if not (0 <= ord(c) <= 127):
                detected_nonascii.add(c)
                nonascii_found = True

                if emoji.is_emoji(c):
                    detected_emojis.add(c)
                    emojis_found = True

        if re.search(WikiTextParser.html_tag_regex, row['text']):
            html_found = True

        if re.search(WikiTextParser.url_regex, row['text']):
            url_found = True

        if re.search(WikiTextParser.number_regex, row['text']):
            numbers_found = True
            detected_numbers = detected_numbers.union( re.findall(WikiTextParser.number_regex, row['text']) )
            # for match in re.finditer(self.number_regex, row['text']):
                # print( row['text'][match.start()-10: match.start()] + "[" + row['text'][match.start() : match.end()] + "]" + row['text'][match.end(): match.end()+10] )

        if nonascii_found:
            nonascii_counter += 1
        if emojis_found:
            emojis_counter += 1
        if html_found:
            html_counter += 1
        if url_found:
            url_counter += 1
        if numbers_found:
            numbers_counter += 1

    detected_numbers = [ num.replace(',', '.') for num in detected_numbers ]
    detected_numbers = [ float(num) for num in detected_numbers ]
    detected_numbers.sort()

    print(f"set of numbers             : {len(detected_numbers)}")
    print(f"set of emojis              : {len(detected_emojis)}")
    print(f"set of nonascii characters : {len(detected_nonascii)}")
    print()
    print(f"nonascii lines to all     : {nonascii_counter / len(corpus) * 100:5.2f}%")
    print(f"emoji lines to all        : {emojis_counter / len(corpus) * 100:5.2f}%")
    print(f"html tags in lines to all : {html_counter / len(corpus) * 100:5.2f}%")
    print(f"urls in lines to all      : {url_counter / len(corpus) * 100:5.2f}%")
    print(f"numbers in lines to all   : {numbers_counter / len(corpus) * 100:5.2f}%")



def find_pattern_examples(file_name: AnyStr, data: pd.DataFrame, pattern, num: int = None, rand = True, window_size = 20):
    sample_num = len(data)
    sample_inds = [i for i in range(0, sample_num)]
    samples_to_print = num if num else sample_num
    samples_left = samples_to_print

    if rand:
        random.shuffle(sample_inds)

    file = open(file_name, "w")
    for ind in sample_inds:
        row = data.iloc[ind]

        if re.search(pattern, row['text']):
            samples_to_print -= 1            

            file.write(f"{samples_left - samples_to_print:3d}) row: {ind:8d} |  ")
            for match in re.finditer(pattern, row['text']):
                
                slice_size = match.end() - match.start()
                left_size = math.floor((window_size - slice_size) / 2) 
                right_size = math.ceil((window_size - slice_size) / 2)

                left_chunk = row['text'][match.start()-left_size : match.start()] 
                middle_chunk = "[" + row['text'][match.start() : match.end()].replace("\n", "") + "]" 
                right_chunk = row['text'][match.end() : match.end()+right_size].replace("\n", "")

                left_chunk = (left_size - len(left_chunk)) * " " + left_chunk
                right_chunk = right_chunk + (right_size - len(right_chunk)) * " " + "  |  "

                file.write( left_chunk + middle_chunk + right_chunk)
            file.write("\n")

        if samples_to_print == 0:
            break
    file.close()


    # for index, row in data.iterrows():
    #     for match in re.finditer(pattern, row['text']):
    #         print( row['text'][match.start()-10: match.start()] + "[" + row['text'][match.start() : match.end()] + "]" + row['text'][match.end(): match.end()+10] )


if __name__ == '__main__':
    data: List[AnyStr] = pickle.load(open(WIKI_DATASET_TRAIN_RAW_PATH, "rb"))

    # data_parser = WikiTextParser(data)
    # data_parser.to_lowercase()
    # data_parser.save_parsed_data("dataset//wikitext_train_prased.pickle")
    
    data_parser = WikiTextParser.from_parsed_data("dataset//wikitext_train_prased.pickle")

    print(data_parser.data.iloc[214829]['text'])
    # data_parser.numbers_to_tokens()
    # data_parser.urls_to_tokens()
    
    # print_html_examples(data_parser.data)
    find_pattern_examples("extracted_patterns//number_samples.txt", data_parser.data, WikiTextParser.number_regex, 300, rand=True, window_size=30)
    # find_pattern_examples("extracted_patterns//html_samples.txt", data_parser.data, WikiTextParser.html_tag_regex, 300, rand=True, window_size=50)
    # find_pattern_examples("extracted_patterns//url_samples.txt", data_parser.data, WikiTextParser.url_regex, 300, rand=True, window_size=80)

    # text_analysis(data_parser.data)
    