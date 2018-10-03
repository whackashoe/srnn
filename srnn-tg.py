#!/usr/bin/env python3
'''
Author: Zeping Yu
Sliced Recurrent Neural Network (SRNN). 
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.
This work is accepted by COLING 2018.
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.
If you have any question, please contact me at zepingyu@foxmail.com.
'''

import pandas as pd
import numpy as np
from console_progressbar import ProgressBar

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense



raw_text = open('tinyshakespeare.txt').read().lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: {}".format(n_chars))
print("Total Vocab: {}".format(n_vocab))

texts = []
next_chars = []

seq_length = 100
pb = ProgressBar(total=100, prefix='Generating Sequences')
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    enc_text = [char_to_int[char] for char in seq_in]
    next_char = char_to_int[seq_out]
    #next_char = seq_out
    #df.append({'text': enc_text, 'next_char': next_char}, ignore_index=True)
    texts.append(seq_in)
    next_chars.append(next_char)
    if i % 10000 == 0:
        pb.print_progress_bar((i / (n_chars - seq_length)) * 100)
print()

df = pd.DataFrame({'text': texts, 'next_char': next_chars})
print("Total Patterns: {}".format(df.shape[0]))


Y = df.next_char.values
Y = to_categorical(Y, num_classes=n_vocab)
X = df.text.values

#set hyper parameters
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT=0.1
NUM_FILTERS = 50
MAX_LEN = 512
Batch_size = 100
EPOCHS = 10

#shuffle the data
indices = np.arange(X.shape[0])
np.random.seed(2018)
np.random.shuffle(indices)
X=X[indices]
Y=Y[indices]

#training set, validation set and testing set
nb_validation_samples_val = int((VALIDATION_SPLIT + TEST_SPLIT) * X.shape[0])
nb_validation_samples_test = int(TEST_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val =  X[-nb_validation_samples_val:-nb_validation_samples_test]
y_val =  Y[-nb_validation_samples_val:-nb_validation_samples_test]
x_test = X[-nb_validation_samples_test:]
y_test = Y[-nb_validation_samples_test:]

print('Tokenizing')
#use tokenizer to build vocab
tokenizer1 = Tokenizer(num_words=MAX_NUM_WORDS, char_level=True)
tokenizer1.fit_on_texts(df.text)
vocab = tokenizer1.word_index
print('Tokenizer fit')
x_train_word_ids = tokenizer1.texts_to_sequences(x_train)
x_test_word_ids = tokenizer1.texts_to_sequences(x_test)
x_val_word_ids = tokenizer1.texts_to_sequences(x_val)
print('Generated word ids')

#pad sequences into the same length
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_LEN)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_LEN)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=MAX_LEN)
print('Padded Sequences')

#slice sequences into many subsequences
pb = ProgressBar(total=100, prefix='Slice Test')
x_test_padded_seqs_split=[]
for i in range(x_test_padded_seqs.shape[0]):
    split1=np.split(x_test_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_test_padded_seqs_split.append(a)
    pb.print_progress_bar((i / x_test_padded_seqs.shape[0]) * 100)
print()

pb = ProgressBar(total=100, prefix='Slice Val')
x_val_padded_seqs_split=[]
for i in range(x_val_padded_seqs.shape[0]):
    split1=np.split(x_val_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_val_padded_seqs_split.append(a)
    pb.print_progress_bar((i / x_val_padded_seqs.shape[0]) * 100)
print()
   
pb = ProgressBar(total=100, prefix='Slice Train')
x_train_padded_seqs_split=[]
for i in range(x_train_padded_seqs.shape[0]):
    split1=np.split(x_train_padded_seqs[i],8)
    a=[]
    for j in range(8):
        s=np.split(split1[j],8)
        a.append(s)
    x_train_padded_seqs_split.append(a)
    pb.print_progress_bar((i / x_train_padded_seqs.shape[0]) * 100)
print()

embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                            EMBEDDING_DIM,
                            weights=[np.random.random((MAX_NUM_WORDS + 1, EMBEDDING_DIM))],
                            input_length=MAX_LEN//64,
                            trainable=True)

#build model
print("Build Model")
input1 = Input(shape=(MAX_LEN//64,), dtype='int32')
embed = embedding_layer(input1)
gru1 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8,MAX_LEN//64,), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed2)
Encoder2 = Model(input2,gru2)

input3 = Input(shape=(8,8,MAX_LEN//64), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = GRU(NUM_FILTERS,recurrent_activation='sigmoid',activation=None,return_sequences=False)(embed3)
preds = Dense(n_vocab, activation='softmax')(gru3)
model = Model(input3, preds)

print(Encoder1.summary())
print(Encoder2.summary())
print(model.summary())

#use adam optimizer
from keras.optimizers import Adam
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

#save the best model on validation set
from keras.callbacks import ModelCheckpoint             
savebestmodel = 'save_model/SRNN(8,2)_yelp2013.h5'
checkpoint = ModelCheckpoint(savebestmodel, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks=[checkpoint] 
             
model.fit(np.array(x_train_padded_seqs_split), y_train, 
          validation_data = (np.array(x_val_padded_seqs_split), y_val),
          epochs = EPOCHS,
          batch_size = Batch_size,
          callbacks = callbacks,
          verbose = 1)

#use the best model to evaluate on test set
from keras.models import load_model
best_model= load_model(savebestmodel)          
print(best_model.evaluate(np.array(x_test_padded_seqs_split),y_test,batch_size=Batch_size))
