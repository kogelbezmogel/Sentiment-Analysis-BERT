from typing import AnyStr, List
from multiprocessing import Pool
from functools import partial

import pandas as pd
import numpy as np 

import emoji
import random
import re
import math
import unicodedata
import time

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

    for _, row in corpus.iterrows():
        for c in row['text']:
    
            if not (0 <= ord(c) <= 127):
                if c not in nonascii_dict.keys():
                    nonascii_dict[c] = 1
                else:
                    nonascii_dict[c] += 1

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


def check_if_nonascii_word(word: AnyStr) -> bool:
    for char in word:
        ucode_category = unicodedata.category(char)
        order = ord(char)
        if order > 127 and ucode_category.startswith('L'):
            return True
    return False


def filter_same_token_repetition(words: List, token: AnyStr) -> List:
    tokens_to_remove = []
    for i, word in enumerate(words[:-1]):
        if word == token and words[i+1] == token:
            tokens_to_remove.append(i+1)

    tokens_to_remove.sort(reverse=True)
    for id in tokens_to_remove:
        words.pop(id)
    return words


def replace_abrevation_in_text(row: pd.Series, abrevations, call_point:int, time_start, loud=False):
    text = row['text']
    current_point = row.name
    words = text.split()

    for abbr in abrevations.keys():
        for i, word in enumerate(words):
            if word == abbr:
                words[i] = abrevations[abbr]
        text = " " + " ".join(words) + " "

    if current_point == call_point and loud:
        time_end = time.time()
        print(f"callpoint time: {time_end - time_start}s")
    
    row['text'] = text
    return row


def replace_abrevation_in_chunk(chunk: pd.DataFrame, abrevations) -> pd.DataFrame:
    old_index = chunk.index
    chunk.reset_index(drop=True, inplace=True)

    first = chunk.iloc[0].name
    last = chunk.iloc[-1].name
    call_point = math.ceil((last - first) * 0.01)
    time_start = time.time()
    
    chunk.apply(partial(replace_abrevation_in_text, abrevations=abrevations, call_point=call_point, time_start=time_start), axis=1)
    chunk.index = old_index
    return chunk



def __chunk_apply(chunk: pd.DataFrame, func, axis, loud=False, **kwargs):
    if loud:
        # resetting index
        index_org = chunk.index
        chunk.reset_index(inplace=True, drop=True)    
        # attributes for measuring progress
        kwargs['chunk_size'] = chunk.iloc[-1].name - chunk.iloc[0].name + 1
        kwargs['call_point'] = math.ceil(0.02 * kwargs['chunk_size'])
        kwargs['start_time'] = time.time()

    chunk.apply(partial(func, **kwargs), axis=axis)

    # restoring index        
    if loud:
        chunk.index = index_org
    return chunk


def __dataframe_split(data: pd.DataFrame, chunks_num: int = None, chunk_size: int = None) -> list[pd.DataFrame]:
    if chunks_num is None and chunk_size is None:
        raise Exception("Number of chunks or chunk size must be given")
    elif chunks_num is not None and chunk_size is not None:
        raise Exception("Both parameters cannot be given in the same time: chunks number and chunk size")
    
    data_size = len(data)
    chunks = []
    if chunks_num:
        chunk_size = math.ceil(data_size / chunks_num)
    elif chunk_size:
        chunks_num = math.ceil(data_size / chunk_size)

    for i in range(chunks_num):
            slice_start = i * chunk_size
            slice_end = min((i+1) * chunk_size, data_size)
            chunks.append(data.iloc[slice_start : slice_end])
    return chunks


def parallel_apply(data: pd.DataFrame, func, axis, n_cores, loud=False, **kwargs) -> pd.DataFrame:
    # This function enables to process dataframe in parallel.
    # func is user defined fuction wihich is given row as first argument and user defined other arguments passed in kwargs. func must return processed row
    # func can be used with loud_decorator by func = partial(loud_decorator, inner_func=func) which prints time estimations for each core.

    w_start = time.time()
    
    pool = Pool(n_cores)
    data_split = __dataframe_split(data, chunks_num=n_cores)

    data_split = pool.map(partial(__chunk_apply, func=func, axis=axis, loud=loud, **kwargs), data_split)
    pool.close()
    pool.join()
    
    data = pd.concat(data_split)
    w_end = time.time()
    print(f"cores: {n_cores}    whole_time: {w_end - w_start:4.0f}s")
    return data


def loud_decorator(row: pd.Series, inner_func, start_time: float, chunk_size: int, call_point: int, **kwargs):
    # It is a decorator that prints aproximated time of job execution after few percent of processed rows
    # inner_func must be second as the loud_dec will be used in apply where row is given as first positional argument
    row = inner_func(row=row, **kwargs)

    if row.name == call_point:
        end_time = time.time()
        print(f"expected processing time: {(end_time - start_time) * chunk_size / call_point:4.0f}s   chunk: {chunk_size} rows   callpoint {call_point} rows   time passed till callpoint {end_time - start_time:4.2f}s")
    return row


if __name__ == "__main__":

    #### test for parralel chunk processing
    
    # # creating random text data with around 200 words per row 
    general_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', ' ']
    def rand_str():
        cp = [ random.choice(general_list) for _ in range( 200 * len(general_list) )]
        return "".join(cp)
    data = [rand_str() for i in range(200_000)]
    data = pd.DataFrame(data, columns=['text'])
    data.to_csv("test_data.csv", header=data.columns, index=False)
    data_cp = pd.read_csv("test_data.csv")
    print(data_cp.info())

    # creating processing function
    def inner(row, abrevations):
        text = row['text']
        words = list(text)

        for abbr in abrevations.keys():
            found_id = []
            for i, word in enumerate(words):
                if word == abbr:
                    found_id.append(i)        
            for id in found_id:
                words[id] = abrevations[abbr]

        row['text'] = "".join(words)
        return row

    # creting argument for processing funciton
    abrevations = {}
    for num, el in enumerate(general_list[:-1]):
        abrevations[el] = str(num)
    print(abrevations)

    # usage
    n_cores = 3
    dec_fun = partial(loud_decorator, inner_func=inner)
    data_cp = pd.read_csv("test_data.csv")
    df = parallel_apply(data_cp, dec_fun, axis=1, n_cores=n_cores, loud=True, abrevations=abrevations)
    print(df)

    data_cp = pd.read_csv("test_data.csv")    
    df = parallel_apply(data_cp, inner, axis=1, n_cores=n_cores, loud=False, abrevations=abrevations)
    print(df)