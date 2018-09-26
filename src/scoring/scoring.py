"""
Scoring methods for ILD capstone.
"""
from rouge import Rouge, FilesRouge
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

d2 = {'pred': [hypothesis,hypothesis], 'actuals': [hypothesis,reference]}
notes2 = pd.DataFrame(data=d2,columns=["pred","actuals"])

def score_data_rouge(df, predicted, actuals):
    rouge = Rouge()
    scorelist = []
    for index,row in df.iterrows():
        rougescore = rouge.get_scores(row[predicted],row[actuals])
        tmpdf = pd.DataFrame(rougescore)
        for row in tmpdf:
            scorelist.append(tmpdf
            [row].apply(pd.Series).values[0])
    score_df = pd.DataFrame(np.array(scorelist).reshape(int(len(scorelist)/3),9),columns=['rouge-1-f','rouge-1-p','rouge-1-r','rouge-2-f','rouge-2-p','rouge-2-r','rouge-l-f','rouge-2-p','rouge-2-r'])
    return score_df


def score_data(method, df, predicted, actuals):
    """
    Applies nlp similarity scores
    :param method - string, scoring type
    :param df - pandas dataframe of predicted and actual text
    :param predicted - string, df column containing predictions
    :param actuals - string, df column containing actuals

    -- if we really care these methods should probably return the same thing
    """
    if method == 'rouge':
        return score_data_rouge(df, predicted, actuals)
    elif method == 'cosine':
        # write cosine code
        df['similarity'] = df.apply(lambda row: cosine_similarity(row, predicted, actuals), axis=1)
        return df
    else:
        print(f'method: {method} undefined')


def cosine_similarity(row, predicted, actuals):
    """
    Accepts a single df row and applies cosine similarity
    """
    spcy = spacy.load('en_core_web_sm') # made need to update this based on what you have locally
    return spcy(row[predicted]).similarity(spcy(row[actuals]))


### Examples
print("hi i'm a rouge")
rougey_rouge = score_data('rouge', notes2, 'pred', 'actuals')
print(rougey_rouge.head())

print("hi i'm a cosine")
cos = score_data('cosine', notes2, 'pred', 'actuals')
print(cos.head())

print("i don't work")
score_data('random scoring', notes2, 'pred', 'actuals')
