#!/usr/bin/env python3
'''
Author: Zeping Yu
Sliced Recurrent Neural Network (SRNN). 
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.
This work is accepted by COLING 2018.
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.
If you have any question, please contact me at zepingyu@foxmail.com.
'''

import argparse
import os
import time

import pandas as pd
import numpy as np
from console_progressbar import ProgressBar

import tensorflow as tf
from tensorflow import keras

from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, CuDNNGRU, TimeDistributed, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint             


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, required=True, help='')
parser.add_argument('--output', type=str, required=True, help='')
parser.add_argument('--max_num_words', type=int, default=256, help='')
parser.add_argument('--embedding-dim', type=int, default=512, help='')
parser.add_argument('--slice-width', type=int, default=8, help='')
parser.add_argument('--validation-split', type=float, default=0.1, help='')
parser.add_argument('--test-split', type=float, default=0.1, help='')
parser.add_argument('--max_len', type=int, default=512, help='')
parser.add_argument('--num-filters', type=int, default=1024, help='')
parser.add_argument('--batch-size', type=int, default=100, help='')
parser.add_argument('--epochs', type=int, default=10, help='')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='')
parser.add_argument('--beta_2', type=float, default=0.999, help='')
parser.add_argument('--epsilon', type=float, default=1e-08, help='')
args = parser.parse_args()


raw_text = open(args.input).read().lower()
n_chars = len(raw_text)
print("Total Characters: {}".format(n_chars))


#use tokenizer to build vocab
tokenizer = Tokenizer(num_words=args.max_num_words, char_level=True)
tokenizer.fit_on_texts(raw_text)
vocab = tokenizer.word_index
n_vocab = len(vocab)
print("Total Vocab: {}".format(n_vocab))

texts = []
next_chars = []

pb = ProgressBar(total=100, prefix='Generating Sequences')
for i in range(0, n_chars - args.max_len, args.max_len):
    texts.append(raw_text[i:i + args.max_len])
    next_chars.append(vocab[raw_text[i + args.max_len]])

    if i % 10000 == 0:
        pb.print_progress_bar((i / (n_chars - args.max_len)) * 100)
pb.print_progress_bar(100)

df = pd.DataFrame({
    'text':      texts,
    'next_char': next_chars
})
print("Total Sequences: {}".format(df.shape[0]))


Y = df.next_char.values-1
Y = to_categorical(Y, num_classes=n_vocab)
X = df.text.values


#shuffle the data
#indices = np.arange(X.shape[0])
#np.random.seed(2018)
#np.random.shuffle(indices)
#X=X[indices]
#Y=Y[indices]

#training set, validation set and testing set
nb_validation_samples_val = int((args.validation_split + args.test_split) * X.shape[0])
nb_validation_samples_test = int(args.test_split * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val =  X[-nb_validation_samples_val:-nb_validation_samples_test]
y_val =  Y[-nb_validation_samples_val:-nb_validation_samples_test]
x_test = X[-nb_validation_samples_test:]
y_test = Y[-nb_validation_samples_test:]

x_train_seqs = tokenizer.texts_to_sequences(x_train)
x_test_seqs = tokenizer.texts_to_sequences(x_test)
x_val_seqs = tokenizer.texts_to_sequences(x_val)
print('Tokenized sequences')

#pad sequences into the same length
x_train_padded_seqs = pad_sequences(x_train_seqs, maxlen=args.max_len)
x_test_padded_seqs = pad_sequences(x_test_seqs, maxlen=args.max_len)
x_val_padded_seqs = pad_sequences(x_val_seqs, maxlen=args.max_len)
print('Padded Sequences')

#slice sequences into many subsequences
def slice_seq(progress_prefix, padded_seqs):
    pb = ProgressBar(total=100, prefix=progress_prefix)
    ret = []

    for i in range(padded_seqs.shape[0]):
        splitted = np.split(padded_seqs[i],8)
        a = []

        for j in range(8):
            b = np.split(splitted[j],8)
            a.append(b)

        ret.append(a)
        if i % 1000 == 0:
            pb.print_progress_bar((i / padded_seqs.shape[0]) * 100)

    pb.print_progress_bar(100)
    return ret

x_test_padded_seqs_split=slice_seq('Slice Test', x_test_padded_seqs)
x_val_padded_seqs_split=slice_seq('Slice Val', x_val_padded_seqs)
x_train_padded_seqs_split=slice_seq('Slice Train', x_train_padded_seqs)
   
embedding_layer = Embedding(args.max_num_words + 1,
                            args.embedding_dim,
                            weights=[np.random.random((args.max_num_words + 1, args.embedding_dim))],
                            input_length=args.max_len//64,
                            trainable=True)

#build model
print("Build Model")

gate_params = {
    'units':                 args.num_filters,
    'return_sequences':      False,
    'return_state':          False,
    'recurrent_initializer': 'glorot_uniform'
}

input1 = Input(shape=(args.max_len//64,), dtype='int32')
embed = embedding_layer(input1)
gru1 = CuDNNGRU(**gate_params)(embed)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8,args.max_len//64,), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = CuDNNGRU(**gate_params)(embed2)
Encoder2 = Model(input2,gru2)

input3 = Input(shape=(8,8,args.max_len//64), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = CuDNNGRU(**gate_params)(embed3)
preds = Dense(n_vocab, activation='softmax')(gru3)
model = Model(input3, preds)

print(Encoder1.summary())
print(Encoder2.summary())
print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(
        lr = args.lr,
        beta_1  = args.beta_1,
        beta_2  = args.beta_2,
        epsilon = args.epsilon
    ),
    metrics=['acc'])

model.fit(
    np.array(x_train_padded_seqs_split), y_train, 
    validation_data = (np.array(x_val_padded_seqs_split), y_val),
    epochs = args.epochs,
    batch_size = args.batch_size,
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.output,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ],
    verbose = 1)

#use the best model to evaluate on test set
best_model= load_model(args.output)          
print(best_model.evaluate(
    np.array(x_test_padded_seqs_split),
    y_test,
    batch_size=args.batch_size
))

pred = model.predict([x_train_padded_seqs_split], verbose=False)[0]
print(pred)
print(sum(star*prob for star, prob in zip([1,2,3,4,5], pred)))
