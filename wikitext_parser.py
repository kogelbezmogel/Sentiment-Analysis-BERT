from typing import List, AnyStr
from nltk.corpus import stopwords

import time
import helper
import pandas as pd
import numpy as np
import pickle 
import re
import emoji
import emot
# import swifter
from multiprocessing import Pool
from functools import partial

from helper import RegexPatterns
from abrevation_dict import AbrevationList

WIKI_DATASET_TRAIN_RAW_PATH = "dataset//core//wikitext_train_raw.pickle"
WIKI_DATASET_TEST_RAW_PATH = "dataset//core//wikitext_test_raw.pickle"
WIKI_DATASET_TRAIN_PARSED_PATH = "dataset//core//wikitext_train_prased.pickle"

REPLACEMENT_MAP = {
    "\u2013" : "-",
    "\u2019" : "'", 
    "\u2212" : "-",
    "\u201d" : "\"",
    "\u201c" : "\"",
    "\xd7"   : "*",
    "\u2044" : "/",
    "\u2026" : "...",
    "\u2032" : "'",
    "\xb7"   : "*",
    "\u2192" : "->"
}


class WikiTextParser:
    def __init__(self, data: List[AnyStr] = []):
        if data:
            self.data = self.__parse_data_to_dataframe(data)


    @classmethod
    def from_parsed_data(cls, path):
        parser =  WikiTextParser()
        parser.data = pickle.load( open(path, 'rb') )
        return parser


    def to_lowercase(self, n_cores: int = 1):
        inner_f = helper.to_lowercase_in_row
        inner_f_dec = partial(helper.loud_decorator, inner_func=inner_f)
        self.data = helper.parallel_apply(self.data, inner_f_dec, axis=1, n_cores=n_cores, loud=True)


    def numbers_to_tokens(self, token = '[NUM]'):
        for index, row in self.data.iterrows():
            row['text'] = re.sub(self.number_regex, token, row['text'])


    def urls_to_tokens(self, token = '[URL]'):
        for index, row in self.data.iterrows():
            row['text'] = re.sub(self.url_regex, token, row['text'])


    def emojis_to_desc(self):
        for ind, row in self.data.iterrows():
            line = emoji.demojize( row['text'] )
            self.data.loc[ind, 'text'] = line


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
            self.data.loc[ind, 'text'] = line


    def extend_abrevations(self, n_cores: int = 1):
        abbr_list = AbrevationList()

        inner_f = helper.replace_abrevations_in_row
        inner_f_dec = partial(helper.loud_decorator, inner_func = inner_f)

        self.data = helper.parallel_apply(self.data, inner_f_dec, axis=1, n_cores=n_cores, loud=True, abrevations=abbr_list)


    def remove_empty_lines(self, loud=False):
        any_letters = re.compile(r'[a-zA-Z]')
        
        for ind, row in self.data.iterrows():
            text: str = row['text']

            if not re.search(any_letters, text):
                self.data.drop(ind, inplace=True)
                
                if loud:
                    print(f"{ind} row was empty: {text}")


    def remove_all_non_ascii(self):
        for ind, row in self.data.iterrows():
            text: str = row['text']

            # filtering character by character
            text = [ char if ord(char) < 128 else ' ' for char in text ]
            text = ''.join(text)

            # removing double spaces
            text = text.split()
            text = " " + " ".join(text) + " "
            self.data[ind, 'text'] = text


    def replace_foreign_words(self, token: AnyStr = "[WRD]"):
        for ind, row in self.data.iterrows():
            text = row["text"]
            text = text.split()

            for num, word in enumerate(text):
                if helper.check_if_nonascii_word(word):
                    text[num] = token
            helper.filter_same_token_repetition(text, token)
            
            text = " " + " ".join(text) + " "
            self.data.loc[ind, "text"] = text


    def replace_choosen_unicodes(self, replacement_map=REPLACEMENT_MAP):
        for ind, row in self.data.iterrows():
            text: str = row['text']
            
            for key in replacement_map.keys():
                text = text.replace(key, replacement_map[key])
            
            self.data[ind, 'text'] = text


    def save_parsed_data(self, path: AnyStr):
        pickle.dump(self.data, open(path, 'wb'))


    def __parse_data_to_dataframe(self, data: List[AnyStr]):
        article_id = -1
        header_id = 0
        line_id = 0
        index_tuples = []
        text_lines = []

        for line in data:
            if re.match(RegexPatterns.article_regex, line):
                article_id += 1
                header_id = 0
                line_id = 0

            elif re.match(RegexPatterns.header_regex, line):
                header_id += 1
                line_id = 0

            elif line != '':
                index_tuples.append((article_id, header_id, line_id))
                text_lines.append(line)
                line_id += 1

        index = pd.MultiIndex.from_tuples(index_tuples, names=(['article', 'part', 'line']))
        return pd.DataFrame(text_lines, columns=['text'], index=index)
    

    def __len__(self):
        return len(self.data)


    # for index, row in data.iterrows():
    #     for match in re.finditer(pattern, row['text']):
    #         print( row['text'][match.start()-10: match.start()] + "[" + row['text'][match.start() : match.end()] + "]" + row['text'][match.end(): match.end()+10] )


    # def rejoin_stopwords(self, stopwords_set = None):
    #     # doesnt't work
    #     words = stopwords.words('english')
        
    #     for ind, row in self.data.iterrows():
    #         text = row['text']
    #         text = text.split()

    #         # rejoining all stop words with pattern (part ' part)
    #         indexes = [i for i, _ in enumerate(text) if text == "'"]
    #         indexes.reverse()

    #         for i in indexes:
    #             potential_stopword = text[i-1] + text[i] + text[i+1]
    #             if potential_stopword in words:
    #                 text[i-i] = potential_stopword
    #                 text.pop(i+1)
    #                 text.pop(i)

    #         self.data.loc[ind, 'text'] =  " " + " ".join(text) + " "


if __name__ == '__main__':
    # data: List[AnyStr] = pickle.load(open(WIKI_DATASET_TRAIN_RAW_PATH, "rb"))
    # data_parser = WikiTextParser(data)
    # data_parser.to_lowercase()
    # data_parser.urls_to_tokens()
    # data_parser.numbers_to_tokens()
    # data_parser.save_parsed_data("dataset//wikitext_train_prased.pickle")
    
    data_parser = WikiTextParser.from_parsed_data("dataset//core//wikitext_train_prased.pickle")
    # for key in REPLACEMENT_MAP.keys():
        # print(f"{key:3s} {key.encode('ascii', 'backslashreplace')} {unicodedata.category(key)}")
    # helper.findcount_nonascii(data_parser.data, "extracted_patterns//nonascii_sorted_by_frequency.txt")

    # print()
    # print(data_parser.data.iloc[900]['text'])
    # data_parser.rejoin_stopwords()
    # data_parser.save_parsed_data("dataset//wikitext_train_prased.pickle")
    # print("stopwords rejoined")
    # print( data_parser.data.iloc[514622]["text"])
    
    # print_html_examples(data_parser.data)
    # find_pattern_examples("extracted_patterns//number_samples.txt", data_parser.data, [WikiTextParser.number_regex], 300, rand=True, window_size=30)
    # find_pattern_examples("extracted_patterns//html_samples.txt", data_parser.data, [WikiTextParser.html_tag_regex], 300, rand=True, window_size=50)
    # find_pattern_examples("extracted_patterns//url_samples.txt", data_parser.data, [WikiTextParser.url_regex], 300, rand=True, window_size=150)

    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2013")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2014")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2019")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\xb0")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2212")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u201d")], 100, rand=True, window_size=200)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u201c")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\xd7")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u02d0")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2044")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u02d0")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u02c8")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2026")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2032")], 100, rand=True, window_size=50)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2018")], 100, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u02bb")], 100, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u30fc")], 100, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\xb7")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u0131")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2033")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u0627")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\xbd")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\xb2")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\xb1")], 50, rand=True, window_size=100)
    helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2192")], 50, rand=True, window_size=100)
    # helper.find_pattern_examples("extracted_patterns//coma_samples.txt", data_parser.data, [re.compile(r"\u2011")], 50, rand=True, window_size=100)

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
    