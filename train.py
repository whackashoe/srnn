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
from tqdm import tqdm
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorboard_batch_monitor import TensorBoardBatchMonitor

from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, CuDNNGRU, GRU, CuDNNGRU, LSTM, CuDNNLSTM, TimeDistributed, Bidirectional, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint             


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, nargs='+', required=True, help='')
parser.add_argument('--output', type=str, required=False, help='')
parser.add_argument('--vocab', type=str, required=False, help='')

parser.add_argument('--max_len', type=int, default=512, help='')
parser.add_argument('--embedding-dim', type=int, default=512, help='')
parser.add_argument('--num-filters', type=int, default=1024, help='')

parser.add_argument('--validation-split', type=float, default=0.1, help='')

parser.add_argument('--slice-width', type=int, default=8, help='')
parser.add_argument('--batch-size', type=int, default=100, help='')
parser.add_argument('--epochs', type=int, default=1, help='')

parser.add_argument('--layer', type=str, default="CUDNNGRU", choices=['GRU', 'LSTM', 'CUDNNGRU', 'CUDNNLSTM'], help='')
parser.add_argument('--recurrent-activation', type=str, default='sigmoid', help='')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='')
parser.add_argument('--beta_2', type=float, default=0.999, help='')
parser.add_argument('--epsilon', type=float, default=1e-08, help='')
args = parser.parse_args()



#use tokenizer to build vocab
tokenizer_path = args.vocab
if tokenizer_path is None:
    tokenizer_path = args.input[0] + ".pickle"

with open(tokenizer_path, 'rb') as t_handle:
    tokenizer = pickle.load(t_handle)

output_path = args.output
if output_path is None:
    output_path = args.input[0] + ".h5"

vocab   = tokenizer.word_index
n_vocab = len(vocab)
print("Total Vocab: {}".format(n_vocab))


embedding_layer = Embedding(n_vocab + 1,
                            args.embedding_dim,
                            weights=[np.random.random((n_vocab + 1, args.embedding_dim))],
                            input_length=args.max_len//64,
                            trainable=True)

#build model
print("Build Model")


def selected_gate(embed):
    cudnn_gate_params = {
        'units':                 args.num_filters
    }

    normal_gate_params = {
        'units':                args.num_filters,
        'recurrent_activation': args.recurrent_activation,
    }

    if args.layer == 'GRU':
        gate = GRU(**normal_gate_params)(embed)
    elif args.layer == 'LSTM':
        gate = LSTM(**normal_gate_params)(embed)
    elif args.layer == 'CUDNNGRU':
        gate = CuDNNGRU(**cudnn_gate_params)(embed)
    elif args.layer == 'CUDNNLSTM':
        gate = CuDNNLSTM(**cudnn_gate_params)(embed)

    return gate


input1   = Input(shape=(args.max_len//64,), dtype='int32')
embed1   = embedding_layer(input1)
gru1     = selected_gate(embed1)
encoder1 = Model(input1, gru1)

input2   = Input(shape=(8, args.max_len//64,), dtype='int32')
embed2   = TimeDistributed(encoder1)(input2)
gru2     = selected_gate(embed2)
encoder2 = Model(input2,gru2)

input3 = Input(shape=(8, 8, args.max_len//64), dtype='int32')
embed3 = TimeDistributed(encoder2)(input3)
gru3   = selected_gate(embed3)

preds = Dense(n_vocab, activation='softmax')(gru3)
model = Model(input3, preds)

print(encoder1.summary())
print(encoder2.summary())
print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(
        lr      = args.lr,
        beta_1  = args.beta_1,
        beta_2  = args.beta_2,
        epsilon = args.epsilon
    ),
    metrics=['acc'])


for in_file in tqdm(args.input):
    for in_offset in tqdm(range(args.max_len)):
        print('Processing {} {}/{}'.format(in_file, in_offset, args.max_len))

        raw_text = open(in_file).read().lower()[in_offset:]
        n_chars = len(raw_text)
        print("Total Characters: {}".format(n_chars))

        texts      = []
        next_chars = []

        for i in tqdm(range(0, n_chars - args.max_len, args.max_len), 'Splitting'):
            texts.append(raw_text[i:i + args.max_len])
            next_chars.append(vocab[raw_text[i + args.max_len]])

        df = pd.DataFrame({
            'text':      texts,
            'next_char': next_chars
        })
        print("Total Sequences: {}".format(df.shape[0]))


        Y = df.next_char.values-1
        Y = to_categorical(Y, num_classes=n_vocab)
        X = df.text.values
        print('X, Y')

        #training set, validation set and testing set
        nb_validation_samples_val = int((args.validation_split) * X.shape[0])

        x_train = X[:-nb_validation_samples_val]
        y_train = Y[:-nb_validation_samples_val]
        x_val =  X[-nb_validation_samples_val:]
        y_val =  Y[-nb_validation_samples_val:]
        print('X Y copy')

        x_train_seqs = tokenizer.texts_to_sequences(x_train)
        x_val_seqs = tokenizer.texts_to_sequences(x_val)
        print('Tokenized sequences')

        #pad sequences into the same length
        x_train_padded_seqs = pad_sequences(x_train_seqs, maxlen=args.max_len)
        x_val_padded_seqs = pad_sequences(x_val_seqs, maxlen=args.max_len)
        print('Padded Sequences')

        #slice sequences into many subsequences
        def slice_seq(progress_prefix, padded_seqs):
            ret = []
            split = np.split
            for i in tqdm(range(padded_seqs.shape[0]), desc=progress_prefix):
                #python3 sucks and makes map slow since we need to convert back to list
                #ret.append([*map(lambda v: split(v, 8), split(padded_seqs[i], 8))]) 
                S = split(padded_seqs[i], 8)
                a = []
                for j in range(8):
                    a.append(split(S[j], 8))
                ret.append(a)
            return ret

        x_val_padded_seqs_split=slice_seq('Slice Val  ', x_val_padded_seqs)
        x_train_padded_seqs_split=slice_seq('Slice Train', x_train_padded_seqs)
           
        model.fit(
            np.array(x_train_padded_seqs_split), y_train, 
            validation_data = (np.array(x_val_padded_seqs_split), y_val),
            epochs = args.epochs,
            batch_size = args.batch_size,
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    output_path,
                    monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max'
                ),
                TensorBoardBatchMonitor(
                    log_dir=os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S")),
                    batch_size=args.batch_size,
                    #histogram_freq=1,
                    #write_graph=True,
                    #write_grads=True,
                    #write_images=True,
                )
            ],
            verbose = 1)

#
##use the best model to evaluate on test set
#best_model= load_model(output_path)          
#print('Evaluate: ', best_model.evaluate(
#    np.array(x_test_padded_seqs_split),
#    y_test,
#    batch_size=args.batch_size
#))
