from rouge import Rouge, FilesRouge
import pandas as pd
import numpy as np

rouge = Rouge()

# read in actual notes
# notes = pd.read_csv("Z:/final_data/scored_data/notes_scored.csv",index_col=0,keep_default_na=False)
notes = pd.read_csv('/gpfs/data/ildproject-share/final_data/scored_data/notes_scored.csv',index_col=0,keep_default_na=False)
notes.columns = ['scored_summaries']

# read in actual df 
# test_dta = pd.read_csv("Z:/final_data/cohorts_merged_test.csv",keep_default_na=False)
test_dta = pd.read_csv("/gpfs/data/ildproject-share/final_data/cohorts_merged_test.csv",keep_default_na=False)

# score_data function
def score_data(df, predicted, actuals):
    printcounter = 0
    for index,row in df.iterrows():
        printcounter += 1
        if(printcounter % 1000)==0:
            print("[info] 1000 iterations")
        if str(row[predicted])!="nan":
            indexlist.append(index)
            rougescore = rouge.get_scores(row[predicted],row[actuals])
            tmpdf = pd.DataFrame(rougescore)
            for row in tmpdf:
                scorelist.append(tmpdf[row].apply(pd.Series).values[0])
        else:
            continue
            
# define average score fx
def avg_rouge_score(df,column_string):
    avg_score_list = []
    avg_score_cols = []
    for col in df.columns:
        if column_string in col:
            print("[info] string found", col)
            avg_score_cols.append(col)
            avgscore = np.nanmean(df[col])
            avg_score_list.append(avgscore)
    tmpdf = pd.DataFrame({
        'avg_score_cols':avg_score_cols,
        'score':avg_score_list
    })
    return(tmpdf)

# get np array now of rouge scores
scorelist = []
indexlist = []
score_data(scored_df,'scored_summaries','findings')
print(len(scorelist))
print(len(indexlist))

# assign
rouged_df = pd.DataFrame(np.array(scorelist).reshape(int(len(scorelist)/3),9),columns=['rouge-1-f','rouge-1-p','rouge-1-r','rouge-2-f','rouge-2-p','rouge-2-r','rouge-l-f','rouge-2-p','rouge-2-r'],index=indexlist)
print(rouged_df.head(n=10))

# merging, scoring
scored_rouged_df = pd.merge(scored_df, rouged_df, left_index=True, right_index=True,how="left")
avg_score_df = avg_rouge_score(scored_rouged_df,"rouge")
print(avg_score_df)

# scored_rouged_df.to_csv("Z:/final_data/scored_data/scored_rouged_df.csv")
scored_rouged_df.to_csv("/gpfs/data/ildproject-share/final_data/scored_data/scored_rouged_df.csv")

# scored date
filename = "/gpfs/data/ildproject-share/final_data/scored_data/avg_score_%s.csv" % (str(datetime.datetime.now()).split('.')[0].replace(' ','_').replace(':','_'))

# avgb score now
avg_score_df.to_csv(filename)
