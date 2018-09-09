#!/usr/bin/env python3

import h5py
import numpy as np

import argparse
import os
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, required=True, help='')
parser.add_argument('--output', type=str, default=(time.strftime("%Y-%m-%d-%H-%M-%S")+".h5"), help='')
parser.add_argument('--max_num_words', type=int, default=30000, help='')
parser.add_argument('--num_filters', type=int, default=50, help='')
parser.add_argument('--max_len', type=int, default=512, help='')
parser.add_argument('--slice_width', type=int, default=8, help='')
parser.add_argument('--batch_size', type=int, default=100, help='')
parser.add_argument('--epochs', type=int, default=10, help='')
parser.add_argument('--recurrent_activation', type=str, default='sigmoid', help='')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='')
parser.add_argument('--beta_2', type=float, default=0.999, help='')
parser.add_argument('--epsilon', type=float, default=1e-08, help='')
parser.add_argument('--layer', type=str, default="GRU", choices=['GRU', 'LSTM', 'CUDNNGRU', 'CUDNNLSTM'], help='')
args = parser.parse_args()

from tensorboard_batch_monitor import TensorBoardBatchMonitor
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Input, Embedding, GRU, CuDNNGRU, LSTM, CuDNNLSTM, TimeDistributed, Bidirectional, Dense
from keras.models import load_model
from keras.optimizers import Adam


with h5py.File(args.dataset, 'r') as hf:
    embedding_matrix = hf['embedding_matrix'][:]
    embedding_dim = len(embedding_matrix[0])
    embedding_layer = Embedding(input_dim=args.max_num_words + 1,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=args.max_len // (args.slice_width ** 2),
                                trainable=True,
                                name='embedding')

    print("Build Model")

    input1 = Input(shape=(args.max_len // (args.slice_width ** 2),), dtype='int32')
    input2 = Input(shape=(
        args.slice_width,
        args.max_len // (args.slice_width ** 2),
    ), dtype='int32')
    input3 = Input(shape=(
        args.slice_width,
        args.slice_width,
        args.max_len // (args.slice_width ** 2)
    ),  dtype='int32')



    if args.layer == 'GRU':
        layer1 = GRU(
            args.num_filters,
            recurrent_activation=args.recurrent_activation,
            activation=None,
            return_sequences=False
        )(embedding_layer(input1))
        Encoder1 = Model(input1, layer1)

        layer2 = GRU(
            args.num_filters,
            recurrent_activation=args.recurrent_activation,
            activation=None,
            return_sequences=False
        )(TimeDistributed(Encoder1)(input2))
        Encoder2 = Model(input2, layer2)

        layer3 = GRU(
            args.num_filters,
            recurrent_activation=args.recurrent_activation,
            activation=None,
            return_sequences=False
        )(TimeDistributed(Encoder2)(input3))
    if args.layer == 'LSTM':
        layer1 = LSTM(
            args.num_filters,
            recurrent_activation=args.recurrent_activation,
            activation=None,
            return_sequences=False
        )(embedding_layer(input1))
        Encoder1 = Model(input1, layer1)

        layer2 = LSTM(
            args.num_filters,
            recurrent_activation=args.recurrent_activation,
            activation=None,
            return_sequences=False
        )(TimeDistributed(Encoder1)(input2))
        Encoder2 = Model(input2, layer2)

        layer3 = LSTM(
            args.num_filters,
            recurrent_activation=args.recurrent_activation,
            activation=None,
            return_sequences=False
        )(TimeDistributed(Encoder2)(input3))
    elif args.layer == 'CUDNNGRU':
        layer1 = CuDNNGRU(
            args.num_filters,
            return_sequences=False
        )(embedding_layer(input1))
        Encoder1 = Model(input1, layer1)

        layer2 = CuDNNGRU(
            args.num_filters,
            return_sequences=False
        )(TimeDistributed(Encoder1)(input2))
        Encoder2 = Model(input2, layer2)

        layer3 = CuDNNGRU(
            args.num_filters,
            return_sequences=False
        )(TimeDistributed(Encoder2)(input3))
    elif args.layer == 'CUDNNLSTM':
        layer1 = CuDNNLSTM(
            args.num_filters,
            return_sequences=False
        )(embedding_layer(input1))
        Encoder1 = Model(input1, layer1)

        layer2 = CuDNNLSTM(
            args.num_filters,
            return_sequences=False
        )(TimeDistributed(Encoder1)(input2))
        Encoder2 = Model(input2, layer2)

        layer3 = CuDNNLSTM(
            args.num_filters,
            return_sequences=False
        )(TimeDistributed(Encoder2)(input3))

    Encoder3 = Dense(5, activation='softmax')(layer3)


    model = Model(input3, Encoder3)


    print(Encoder1.summary())
    print(Encoder2.summary())
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(
                      lr       = args.lr,
                      beta_1   = args.beta_1,
                      beta_2   = args.beta_2,
                      epsilon  = args.epsilon
                  ),
                  metrics=['acc'])

    #save the best model on validation set
    bestmodel_path = 'save_model/' + args.output
    checkpoint = ModelCheckpoint(
        bestmodel_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    tensorboard = TensorBoardBatchMonitor(
        log_dir=os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S")),
        histogram_freq=1,
        batch_size=args.batch_size,
        write_graph=True,
        write_grads=True,
        write_images=True,
        embeddings_freq=args.batch_size,
        embeddings_layer_names=['embedding'],
        embeddings_metadata=None,
        embeddings_data=None
    )
    tensorboard.set_model(model)

    callbacks=[checkpoint, tensorboard]

    model.fit(np.array(hf['x_train_padded_seqs_split'][:]), hf['y_train'][:],
              validation_data = (np.array(hf['x_val_padded_seqs_split'][:]), hf['y_val'][:]),
              epochs     = args.epochs,
              batch_size = args.batch_size,
              callbacks  = [checkpoint, tensorboard],
              verbose    = 1)

    #use the best model to evaluate on test set
    best_model = load_model(bestmodel_path)
    print(best_model.evaluate(
        np.array(hf['x_test_padded_seqs_split'][:]),
        hf['y_test'][:],
        batch_size=args.batch_size
    ))
