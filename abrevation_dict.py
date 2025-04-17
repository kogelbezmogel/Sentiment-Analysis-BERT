# List source
# Deveci Kocakoç,I. Abbreviation List for NLP studies, (2021),
# GitHub repository, https://github.com/ipekdk/abbreviation-list-english


import pandas as pd


ABREVATIONS_PATH = "files//abrevations_eng.csv"

class AbrevationList:
    
    def __init__(self):
        self.data = pd.read_csv(ABREVATIONS_PATH, encoding='latin-1', sep=';')
        self.data['abbr'] = self.data['abbr'].astype(str).apply(lambda value: value.lower())
        self.data['long'] = self.data['long'].astype(str).apply(lambda value: value.lower().replace("õ", "'"))

        self.data.drop('no', axis=1, inplace=True)
        self.data.set_index('abbr', inplace=True)
        print(self.data)

    def keys(self):
        return set(self.data.index.to_list())

    def __getitem__(self, key):
        return self.data.loc[key]['long']
