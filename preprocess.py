#!/usr/bin/env python3

'''
Author: Zeping Yu
Sliced Recurrent Neural Network (SRNN).
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.
This work is accepted by COLING 2018.
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.
If you have any question, please contact me at zepingyu@foxmail.com.
'''

import sys

import argparse
import pandas as pd
import numpy as np
import h5py

from util import seqs_split

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv', type=str, required=True, help='')
parser.add_argument('--output', type=str, required=True, help='')
parser.add_argument('--glove_path', type=str, default='glove/glove.6B.200d.txt', help='')
parser.add_argument('--max_num_words', type=int, default=30000, help='')
parser.add_argument('--slice_width', type=int, default=8, help='')
parser.add_argument('--validation_split', type=float, default=0.1, help='')
parser.add_argument('--test_split', type=float, default=0.1, help='')
parser.add_argument('--max_len', type=int, default=512, help='')
parser.add_argument('--embedding_dim', type=int, default=200, help='')
args = parser.parse_args()



from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

with h5py.File(args.output, 'w') as hf:
    df = pd.read_csv(args.csv)
    print('{} read'.format(args.csv))
    #df = df.sample(5000)

    Y = df.stars.values-1
    Y = to_categorical(Y,num_classes=5)
    X = df.text.values


    #shuffle the data
    indices = np.arange(X.shape[0])
    np.random.seed(2018)
    np.random.shuffle(indices)
    X=X[indices]
    Y=Y[indices]

    #training set, validation set and testing set
    nb_validation_samples_val = int((args.validation_split + args.test_split) * X.shape[0])
    nb_validation_samples_test = int(args.test_split * X.shape[0])

    x_train = X[:-nb_validation_samples_val]
    y_train = Y[:-nb_validation_samples_val]
    x_val =  X[-nb_validation_samples_val:-nb_validation_samples_test]
    y_val =  Y[-nb_validation_samples_val:-nb_validation_samples_test]
    x_test = X[-nb_validation_samples_test:]
    y_test = Y[-nb_validation_samples_test:]
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('y_val',   data=y_val)
    hf.create_dataset('y_test',  data=y_test)

    #use tokenizer to build vocab
    tokenizer = Tokenizer(num_words=args.max_num_words)
    tokenizer.fit_on_texts(df.text)
    vocab = tokenizer.word_index
    print('tokenizer fit_on_texts done')

    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    print('x_train_word_ids texts_to_sequences done')
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    print('x_test_word_ids texts_to_sequences done')
    x_val_word_ids = tokenizer.texts_to_sequences(x_val)
    print('x_val_word_ids texts_to_sequences done')

    #pad sequences into the same length
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=args.max_len)
    print('x_train_padded_seqs pad_sequences done')
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=args.max_len)
    print('x_test_padded_seqs pad_sequences done')
    x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=args.max_len)
    print('x_val_padded_seqs pad_sequences done')

    x_test_padded_seqs_split = seqs_split(x_test_padded_seqs, args.slice_width)
    hf.create_dataset('x_test_padded_seqs_split', data=x_test_padded_seqs_split)
    print('x_test_padded_seqs_split done')

    x_val_padded_seqs_split = seqs_split(x_val_padded_seqs, args.slice_width)
    hf.create_dataset('x_val_padded_seqs_split', data=x_val_padded_seqs_split)
    print('x_val_padded_seqs_split done')

    x_train_padded_seqs_split = seqs_split(x_train_padded_seqs, args.slice_width)
    hf.create_dataset('x_train_padded_seqs_split', data=x_train_padded_seqs_split)
    print('x_train_padded_seqs_split done')

    print("Using GloVe embeddings")
    embeddings_index = {}
    f = open(args.glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    #use pre-trained GloVe word embeddings to initialize the embedding layer
    embedding_matrix = np.random.random((args.max_num_words + 1, args.embedding_dim))
    for word, i in vocab.items():
        if i<args.max_num_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be random initialized.
                embedding_matrix[i] = embedding_vector

    hf.create_dataset('embedding_matrix', data=embedding_matrix)
