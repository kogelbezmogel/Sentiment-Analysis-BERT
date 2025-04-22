from typing import AnyStr, List
import pandas as pd


MAIN_FOLDER_PATH = "dataset//sentiment"

class AmazonSentiment:
    DATASET_PATH = "dataset//sentiment//amazon_reviews_sentiment//train.ft.txt.bz2"

    def __init__(self):
        self.data = pd.read_csv(self.DATASET_PATH, compression='bz2', delimiter='\t', header=None)
        self.data.columns = ['raw']

        self.data['raw'] = self.data['raw'].apply(lambda line: line[9:10] + " ##split##" + line[10:] )
        self.data[['label', 'text']] = self.data['raw'].str.split("##split##", expand=True)
        self.data.drop('raw', axis=1, inplace=True)

        self.data['label'] = self.data['label'].apply(lambda label: 'positive' if int(label) == 2 else 'negative')
        print(self.data)


class DrugsSentiment:
    DATASET_PATH = "dataset//sentiment//"
    def __init__(self):
        pass


class SocialMediaSentiment:
    DATASET_PATH = ""
    def __init__(self):
        pass


class AirlineSentiment:
    DATASET_PATH = ""
    def __init__(self):
        pass


class MoviesSentiment:
    DATASET_PATH = ""
    def __init__(self):
        pass


class SentimentDatasetsParser:
    def __init__(self, datasets: List):
        pass

    

if __name__ == "__main__":
    data = AmazonSentiment()