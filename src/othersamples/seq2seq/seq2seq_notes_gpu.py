import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
import os
import datetime
import glob

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print('TensorFlow Version: {}'.format(tf.__version__))

os.getcwd()

# notes = pd.read_csv("Z:/notes v2/cohort1_deid_df.csv")
# notes = pd.read_csv("Z:/final_data/cohort1_final_data.csv")
notes = pd.read_csv("/gpfs/data/ildproject-share/final_data/cohorts_merged_training.csv")

notes = notes[['findings','impressions']]
# notes = notes.head(n=100000)
notes.head()
notes.isnull().sum()

# Remove null values and unneeded features
notes = notes.dropna()
# notes = notes.drop(['Unnamed: 0','ScanType','ScanDate','ClinInfo','Technique','Comparison',
#                         'ElecSig','Patient','RawText'], 1)
notes = notes.reset_index(drop=True)
# notes.head()

print("[info] dimensions of notes",notes.shape)
print("[info] length of unique findings",len(np.unique(notes.findings)))

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return(text)

# remove stopwords, keepo for summary
# Clean the summaries and texts
clean_summaries = []
for summary in notes.impressions:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in notes.findings:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)

print("Size of Vocabulary:", len(word_counts))

# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better
# (https://github.com/commonsense/conceptnet-numberbatch)
embeddings_index = {}
# with open('Z:/helper/numberbatch-en-17.06.txt', encoding='utf-8') as f:
with open('/gpfs/data/ildproject-share/helper/numberbatch-en-17.06.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 10

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words/len(word_counts),4)*100

print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers
vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())

def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count

max_length_tmp = 1
all_lengths_tmp = []
for length in lengths_texts.counts:
#     print(length)
    all_lengths_tmp.append(length)
    if length > max_length_tmp:
        max_length_tmp = length

print("[info] max length: ", max_length_tmp)
print("[info] avg length: ", np.mean(all_lengths_tmp))

max_length_tmp = 1
all_lengths_tmp = []
for length in lengths_summaries.counts:
#     print(length)
    all_lengths_tmp.append(length)
    if length > max_length_tmp:
        max_length_tmp = length

print("[info] max length summaries: ", max_length_tmp)
print("[info] avg length summaries: ", np.mean(all_lengths_tmp))

# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove notes that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 1220
max_summary_length = 294
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length):
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
            len(int_texts[count]) >= min_length and
            unk_counter(int_summaries[count]) <= unk_summary_limit and
            unk_counter(int_texts[count]) <= unk_text_limit and
            length == len(int_texts[count])
           ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])

# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))

def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')
    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

def process_encoding_input(target_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1]) # slice it to target_data[0:batch_size, 0: -1]
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
            enc_output = tf.concat(enc_output,2)
            # original code is missing this line below, that is how we connect layers
            # by feeding the current layer's output to next layer's input
            rnn_inputs = enc_output
    return enc_output, enc_state

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, output_layer,
                            vocab_size, max_summary_length,batch_size):
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                       helper=training_helper,
                                                       initial_state=dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                       output_layer = output_layer)

    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_summary_length)
    return training_logits

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, output_layer,
                             max_summary_length, batch_size):
    '''Create the inference logits'''

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                        output_layer)

    inference_logits = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_summary_length)

    return inference_logits

def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    dec_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size, keep_prob) for _ in range(num_layers)])
    output_layer = Dense(vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     text_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,attn_mech,rnn_size)
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,summary_length,dec_cell,
                                                  output_layer,
                                                  vocab_size,
                                                  max_summary_length,
                                                  batch_size)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)
    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''

    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size) #shape=(batch_size, senquence length) each seq start with index of<GO>
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    training_logits, inference_logits  = decoding_layer(dec_embed_input,
                                                        embeddings,
                                                        enc_output,
                                                        enc_state,
                                                        vocab_size,
                                                        text_length,
                                                        summary_length,
                                                        max_summary_length,
                                                        rnn_size,
                                                        vocab_to_int,
                                                        keep_prob,
                                                        batch_size,
                                                        num_layers)
    return training_logits, inference_logits

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

print("'<PAD>' has id: {}".format(vocab_to_int['<PAD>']))
sorted_summaries_samples = sorted_summaries[7:50]
sorted_texts_samples = sorted_texts[7:50]
pad_summaries_batch_samples, pad_texts_batch_samples, pad_summaries_lengths_samples, pad_texts_lengths_samples = next(get_batches(
    sorted_summaries_samples, sorted_texts_samples, 5))
print("pad summaries batch samples:\n\r {}".format(pad_summaries_batch_samples))

# Set the Hyperparameters
epochs = 1
# batch_size = 64
# batch_size = 10
batch_size = 100
rnn_size = 256
num_layers = 3
learning_rate = 0.005
keep_probability = 0.90

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():

    # Load the model inputs
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits[0].rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits[0].sample_id, name='predictions')

    # Create the weights for sequence_loss, the sould be all True across since each batch is padded
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")
graph_location = "./graph"
print(graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(train_graph)

# Subset the data for training
# start = 200000
start = 0
end = start + len(sorted_summaries)
# end = start + 50000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))

ckpt_text = "/gpfs/data/ildproject-share/modelparams/seq2seq/best_model_%s.ckpt" % (str(datetime.datetime.now()).split('.')[0].replace(' ','_').replace(':','_'))
print(ckpt_text)

# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 1 # Check training loss after every 20 batches
stop_early = 0
stop = 1 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1

update_loss = 0
batch_loss = 0
summary_update_loss = [] # Record the update losses for saving improvements in the model

# ckpt_text = "./best_model.ckpt"
checkpoint = ckpt_text
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=train_graph, config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # If we want to continue training a previous session
    #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
    #loader.restore(sess, checkpoint)

    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 summary_length: summaries_lengths,
                 text_length: texts_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(sorted_texts_short) // batch_size,
                              batch_loss / display_step,
                              batch_time*display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)

                # If the update loss is at a new minimum, save the model
                if update_loss <= min(summary_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0


        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate

        if stop_early == stop:
            print("Stopping Training.")
            break

## Scoring section ##
print("[info] scoring section")

# input_sentences=["Increase in diffuse pulmonary opacities which may represent pulmonary edema, pulmonary hemorrhage, pulmonary alveolar proteinosis, and/or atypical infection",
#                "Diffuse nonspecific interstitial disease of uncertain etiology, without appreciable interval change"]
# generagte_summary_length =  [3,2]

# test_texts = [text_to_seq(input_sentence) for input_sentence in input_sentences]

#### Now we move onto the next section
print("[info] now we will score the test set")

# get max file
list_of_files = glob.glob('/gpfs/data/ildproject-share/modelparams/seq2seq/*.meta') # * means all if need specific format then *.csv
# list_of_files = glob.glob('Z:/modelparams/seq2seq/*.meta') # * means all if need specific format then *.csv
print("[info] list of files from training", list_of_files)
# ckpt_text = max(list_of_files, key=os.path.getctime)
# ckpt_text = ckpt_text[:-5]

print("[info] the max file by time is: ",ckpt_text)

def text_to_seq(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]

# load in test data set
# test_notes = pd.read_csv("Z:/final_data/cohorts_merged_test.csv")
test_notes = pd.read_csv("/gpfs/data/ildproject-share/final_data/cohorts_merged_test.csv")
test_notes = test_notes[['findings','impressions']]
test_notes = test_notes.dropna()

# only if we need to test the script do we subset below
# test_notes = test_notes.head(n=50)
print(test_notes.head())

# test_notes = test_notes.reset_index(drop=True)

print("[info] dimensions of notes",test_notes.shape)

# cleaned test data set

# test_clean_summaries = []
# for summary in test_notes.impressions:
#     test_clean_summaries.append(clean_text(summary, remove_stopwords=False))
# print("Summaries are complete.")

# test_clean_texts = []
# for text in test_notes.findings:
#     test_clean_texts.append(clean_text(text))
# print("Texts are complete.")

# len(test_clean_summaries)

# original raw text to sequence for predictions
print("[info] now converting raw text to seq")
test_texts = [text_to_seq(input_sentence) for input_sentence in test_notes.findings]

# original raw text input
print("[info] storing original raw texts")
input_sentences = test_notes.findings

# define the original raw text inputs to join in later our scored summaries.
scans_output = pd.DataFrame(test_notes.findings)

# load in
checkpoint = ckpt_text
summaries_list = []
# define summary length
generagte_summary_length =  500

# actual scoring
if type(generagte_summary_length) is list:
    if len(input_sentences)!=len(generagte_summary_length):
        raise Exception("[Error] makeSummaries parameter generagte_summary_length must be same length as input_sentences or an integer")
    generagte_summary_length_list = generagte_summary_length
else:
    generagte_summary_length_list = [generagte_summary_length] * len(test_texts)
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
    for i, text in enumerate(test_texts):
        generagte_summary_length = generagte_summary_length_list[i]
        answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                          summary_length: [generagte_summary_length], #summary_length: [np.random.randint(5,8)],
                                          text_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0]
        # print(answer_logits)
        # Remove the padding from the summaries
        pad = vocab_to_int["<PAD>"]
        summaries_list.append([int_to_vocab[i] for i in answer_logits if i != pad])
#         print('- Review:\n\r {}'.format(input_sentences[i]))
#         print('- Summary:\n\r {}\n\r\n\r'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))

# now we have the scored
print("[info] input sentences scored and summaries generated")

# print(scans_output)
# print(summaries_list)

# now let's join the summaries list and the input_sentences
scans_output['impressions'] = summaries_list
scans_output['ref_impressions'] = test_notes.impressions

filename = "/gpfs/data/ildproject-share/final_data/scored_data/notes_scored_%s.csv" % (str(datetime.datetime.now()).split('.')[0].replace(' ','_').replace(':','_'))

scans_output.to_csv(filename)