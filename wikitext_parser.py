from typing import List, AnyStr
import emot.core
from nltk.corpus import stopwords

import time
import pandas as pd
import pickle 
import re
import emoji
import emot
import random
import math

from abrevation_dict import AbrevationList

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


    def emojis_to_desc(self):
        for ind, row in self.data.iterrows():
            line = emoji.demojize( row['text'] )
            data.loc[ind, 'text'] = line


    def emoticons_to_desc(self):
        emot_obj = emot.emot()

        for ind, row in self.data.iterrows():
            line = row['text']
            res = emot_obj.emoticons(line)
        
            locations = res['location']
            descriptions = res['mean']
            locations.reverse()
            descriptions.reverse()           
        
            for i, location in enumerate(locations):
                description = re.sub(r'[^a-zA-Z\d]+', "_", descriptions[i].lower().strip())
                description = ":" + description.replace("_andry_", "_angry_")
                line = line[ : location[0]] + description + line[location[1] : ]        
            data.loc[ind, 'text'] = line


    def extend_abrevations(self):
        abrevations = AbrevationList()
        for abbr in abrevations.keys():
                pattern = re.compile(r" " + re.escape(abbr) + r" ")
                self.data['text'] = self.data['text'].apply(lambda text: re.sub(pattern, f" {abrevations[abbr]} ", text))
    

    def remove_empty_lines(self):
        pass


    def rejoin_stopwords(self, stopwords_set = None):
        # doesnt't work
        words = stopwords.words('english')
        
        for ind, row in self.data.iterrows():
            text = row['text']
            text = text.split()

            # rejoining all stop words with pattern (part ' part)
            indexes = [i for i, _ in enumerate(text) if text == "'"]
            indexes.reverse()

            for i in indexes:
                potential_stopword = text[i-1] + text[i] + text[i+1]
                if potential_stopword in words:
                    text[i-i] = potential_stopword
                    text.pop(i+1)
                    text.pop(i)

            self.data.loc[ind, 'text'] =  " " + " ".join(text) + " "


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
    emptylines_counter = 0

    for index, row in corpus.iterrows():
        nonascii_found = False
        url_found = False
        html_found = False
        emojis_found = False
        numbers_found = False
        emptylines_found = False

        # non-ascii and emojis
        for c in row['text']:

            if not (0 <= ord(c) <= 127):
                detected_nonascii.add(c)
                nonascii_found = True

                if emoji.is_emoji(c):
                    detected_emojis.add(c)
                    emojis_found = True

        if row['text'] == "":
            emptylines_counter += 1
        if re.search(WikiTextParser.html_tag_regex, row['text']):
            html_found = True
        if re.search(WikiTextParser.url_regex, row['text']):
            url_found = True
        if re.search(WikiTextParser.number_regex, row['text']):
            numbers_found = True
            detected_numbers = detected_numbers.union( re.findall(WikiTextParser.number_regex, row['text']) )

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
        if emptylines_found:
            emptylines_counter += 1

    print(f"set of numbers             : {len(detected_numbers)}")
    print(f"set of emojis              : {len(detected_emojis)}")
    print(f"set of nonascii characters : {len(detected_nonascii)}")
    print()
    print(f"nonascii lines to all     : {nonascii_counter / len(corpus) * 100:5.2f}%")
    print(f"emoji lines to all        : {emojis_counter / len(corpus) * 100:5.2f}%")
    print(f"html tags in lines to all : {html_counter / len(corpus) * 100:5.2f}%")
    print(f"urls in lines to all      : {url_counter / len(corpus) * 100:5.2f}%")
    print(f"numbers in lines to all   : {numbers_counter / len(corpus) * 100:5.2f}%")
    print(f"empty lines to all        : {emptylines_counter / len(corpus) * 100:5.2f}%")


def find_pattern_examples(file_name: AnyStr, data: pd.DataFrame, patterns: List, num: int = None, rand = True, window_size = 20):
    file = open(file_name, "w")
    
    sample_num = len(data)
    sample_inds = [i for i in range(0, sample_num)]

    if rand:
        random.shuffle(sample_inds)

    for pattern in patterns:
        samples_to_print = num if num else sample_num
        samples_left = samples_to_print

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

    data_parser = WikiTextParser(data)
    # text_analysis(data_parser.data)
    data_parser.to_lowercase()
    data_parser.urls_to_tokens()
    data_parser.numbers_to_tokens()
    data_parser.save_parsed_data("dataset//wikitext_train_prased.pickle")
    
    # data_parser = WikiTextParser.from_parsed_data("dataset//wikitext_train_prased.pickle")
    # data_parser.rejoin_stopwords()
    # data_parser.save_parsed_data("dataset//wikitext_train_prased.pickle")
    # print("stopwords rejoined")
    # print( data_parser.data.iloc[514622]["text"])
    
    # print_html_examples(data_parser.data)
    # find_pattern_examples("extracted_patterns//number_samples.txt", data_parser.data, [WikiTextParser.number_regex], 300, rand=True, window_size=30)
    # find_pattern_examples("extracted_patterns//html_samples.txt", data_parser.data, [WikiTextParser.html_tag_regex], 300, rand=True, window_size=50)
    # find_pattern_examples("extracted_patterns//url_samples.txt", data_parser.data, [WikiTextParser.url_regex], 300, rand=True, window_size=150)
    # find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r" '\S+")], 20, rand=True, window_size=30)

    # timer_start = time.time()
    # counter = dict()
    # words = stopwords.words("english")
    # for word in words:
    #     counter[word] = 0

    # print(f"{0:2d}% time elapsed: {0:3d} min {0:2d} s")
    # rows_num = len(data_parser.data)

    # i = 0
    # for ind, line in data_parser.data.iterrows():
    #     text = line['text'].split()
    #     for word in words:
    #         counter[word] += text.count(word)
    #     if i % 50 == 0:
    #         timer_end = time.time()
    #         elapsed_time = int(timer_end - timer_end)
    #         print ("\033[A                             \033[A")
    #         print(f"{int(round(100 * i / rows_num, 0)):2d}% time elapsed: {elapsed_time//60:3d} min {elapsed_time%60:2d} s")
    #     i += 1

    # pickle.dump(counter, open("rejoined_stopwords_counter.pickle", "wb"))

    # counter_rejoi = pickle.load(open("files//rejoined_stopwords_counter.pickle", "rb"))
    # counter_clean = pickle.load(open("files//clean_stopwords_counter.pickle", "rb"))

    # for key in counter_clean.keys():
    #     print( f"{key:20s} : {counter_clean[key]:5d} {counter_rejoi[key]:5d}" )

    # text_analysis(data_parser.data)
    