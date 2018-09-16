import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
import glob
import os
import datetime

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print('TensorFlow Version: {}'.format(tf.__version__))

# function
def text_to_seq(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]

    test_notes = pd.read_csv("Z:/final_data/cohort1_final_data.csv")
    # notes = pd.read_csv("gpfs/data/ildproject-share/final_data/cohort1_final_data.csv")

# load data
notes = pd.read_csv("/gpfs/data/ildproject-share/final_data/cohorts_merged_training.csv")

test_notes = test_notes[['findings','impressions']]
test_notes.head()

# null count
test_notes.isnull().sum()

# Remove null values and unneeded features
test_notes = test_notes.dropna()
test_notes = test_notes.reset_index(drop=True)

print("[info] dimensions of notes",test_notes.shape)

print("[info] ", len(test_clean_summaries))

# run summaries
test_texts = [text_to_seq(input_sentence) for input_sentence in test_notes.impressions]
texts = test_texts[0:5]

# run impressions through
input_sentences = test_notes.impressions[0:5]
input_sentences

# summary length
generagte_summary_length =  100;

# get max file
list_of_files = glob.glob('/gpfs/data/ildproject-share/modelparams/seq2seq/*') # * means all if need specific format then *.csv
ckpt_text = max(list_of_files, key=os.path.getctime)
print(ckpt_text)

# now score
checkpoint = ckpt_text
if type(generagte_summary_length) is list:
    if len(input_sentences)!=len(generagte_summary_length):
        raise Exception("[Error] makeSummaries parameter generagte_summary_length must be same length as input_sentences or an integer")
    generagte_summary_length_list = generagte_summary_length
else:
    generagte_summary_length_list = [generagte_summary_length] * len(texts)
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    #Multiply by batch_size to match the model's input parameters
    for i, text in enumerate(texts):
        generagte_summary_length = generagte_summary_length_list[i]
        answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                          summary_length: [generagte_summary_length], #summary_length: [np.random.randint(5,8)],
                                          text_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0]
        print(answer_logits)
        # Remove the padding from the summaries
        pad = vocab_to_int["<PAD>"]
        # print('- Review:\n\r {}'.format(input_sentences[i]))
        # print('- Summary:\n\r {}\n\r\n\r'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
