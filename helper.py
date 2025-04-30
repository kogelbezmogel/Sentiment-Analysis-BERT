from typing import AnyStr, List

import pandas as pd
import emoji
import random
import re
import math


class RegexPatterns:
    header_regex = re.compile(r" = .+ = .*")
    article_regex = re.compile(r"( = ){1}[^=;]+( = ){1}")
    html_tag_regex = re.compile(r'<.*?>')
    url_regex = re.compile(r'(http : / / )?www\.\S+( / \S+)*( \? \S+ = \S+( \& \S+ = \S+)*)?')
    number_regex = re.compile(r'(\d+( @[\.,]@ \d+)+)|(\d+([\.,]\d+)*)')


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

    for _, row in corpus.iterrows():
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
                if emoji.is_emoji(c):
                    detected_emojis.add(c)
                    emojis_found = True
                nonascii_found = True

        if row['text'] == "":
            emptylines_counter += 1
        if re.search(RegexPatterns.html_tag_regex, row['text']):
            html_found = True
        if re.search(RegexPatterns.url_regex, row['text']):
            url_found = True
        if re.search(RegexPatterns.number_regex, row['text']):
            numbers_found = True
            detected_numbers = detected_numbers.union( re.findall(RegexPatterns.number_regex, row['text']) )

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

    print(detected_nonascii)
    print()
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


def findcount_nonascii(corpus: pd.DataFrame, save_path: AnyStr):
    nonascii_dict = dict()

    i = 1
    for _, row in corpus.iterrows():
        for c in row['text']:
    
            if not (0 <= ord(c) <= 127):
                if c not in nonascii_dict.keys():
                    nonascii_dict[c] = 1
                else:
                    nonascii_dict[c] += 1
        # if i > 1000:
        #     break
        # i += 1

    index = nonascii_dict.keys()
    unicode = [ val.encode('ascii', 'backslashreplace') for val in index ]
    values = [nonascii_dict[key] for key in index]
    results = pd.DataFrame(values, columns=['count'], index=index)
    results['unicode'] = unicode

    results.sort_values('count', inplace=True, ascending=False)    
    print(results.head(30))
    
    # saving to file
    file = open(save_path, 'w')
    for ind, row in results.iterrows():
        file.write(f"{row['count']:10d} | {ind:3s} | {row['unicode']}\n")
    file.close()


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


