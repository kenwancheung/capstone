from rouge import Rouge, FilesRouge
import pandas as pd
import numpy as np

# hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
# reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

rouge = Rouge()
# scores = rouge.get_scores(hypothesis, reference)

def score_data(df, predicted, actuals):
    for index,row in df.iterrows():
#         print(row)
        print("[info] Predicted")
        print(row[predicted])
        print("[info] Predicted")
        print(row[actuals])
        rougescore = rouge.get_scores(row[predicted],row[actuals])
        tmpdf = pd.DataFrame(rougescore)
        print("[info] now appending to list with reshape")
        for row in tmpdf:
            scorelist.append(tmpdf[row].apply(pd.Series).values[0])

# get np array now of rouge scores
scorelist = []
score_data(notes2,'pred','actuals')

# assign
score_df = pd.DataFrame(np.array(scorelist).reshape(int(len(scorelist)/3),9),columns=['rouge-1-f','rouge-1-p','rouge-1-r','rouge-2-f','rouge-2-p','rouge-2-r','rouge-l-f','rouge-2-p','rouge-2-r'])
score_df.head()
