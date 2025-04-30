from typing import AnyStr, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helper

MAIN_FOLDER_PATH = "dataset//sentiment"


class SentimentData:
    def __init__(self):
        pass

    def _text_separation(self, data: pd.DataFrame) -> pd.DataFrame:
        
        for ind, row in data.iterrows():
            text = row['text']
            # barker 's
            # 960 , and
            # croydon . she
            # it st. andrew
            # 10 @,@ 472
            # 7 @.@ 5
            # austro @-@ hungarian 
            # 423 kw
            # km / h ; 23 mph
            # temple ( cave ) .
            #  [ has ] a
            # that " this cave [ has ] a beautiful gate "
            data.loc[ind, 'text'] = text


        return data


class AmazonSentiment:
    DATASET_PATH = "dataset//sentiment//amazon_reviews_sentiment//train.ft.txt.bz2"

    def __init__(self):
        self.data = pd.read_csv(self.DATASET_PATH, compression='bz2', delimiter='\t', header=None)
        self.data.columns = ['raw']

        self.data['raw'] = self.data['raw'].apply(lambda line: line[9:10] + " ##split##" + line[10:] )
        self.data[['label', 'text']] = self.data['raw'].str.split("##split##", expand=True)
        self.data.drop('raw', axis=1, inplace=True)

        self.data['label'] = self.data['label'].apply(lambda label: 'positive' if int(label) == 2 else 'negative')
        

class DrugsSentiment(SentimentData):
    DATASET_TRAIN_PATH = "dataset//sentiment//drug_reviews_sentiment//drug_review_train.csv"
    DATASET_VALIDATION_PATH = "dataset//sentiment//drug_reviews_sentiment//drug_review_validation.csv"

    def __init__(self):
        train_part = pd.read_csv(self.DATASET_TRAIN_PATH, delimiter=',')
        validation_part = pd.read_csv(self.DATASET_VALIDATION_PATH, delimiter=',')
        
        self.data = pd.concat([train_part, validation_part], axis=0)
        self.data = self.data[ (self.data['rating'] > 8.0) | (self.data['rating'] < 5.0) ]
        self.data = self.data[['review', 'rating']]
        self.data.columns = ['text', 'label']
        self.data['label'] = self.data['label'].apply(lambda rating: 'positive' if rating > 8 else 'negative')
        
        # text preprocessing
        # self.data = self._text_separation(self.data)
        # print(self.data)


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
    # data = AmazonSentiment()
    data = DrugsSentiment()
    print()
    print( data.data.iloc[0]['text'] )
    # helper.text_analysis(data.data)