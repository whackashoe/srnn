#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tqdm import tqdm

import argparse
import os
import time
import sys
from collections import Counter


from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, required=True, help='')
parser.add_argument('--vocab', type=str, required=True, help='')
parser.add_argument('--text', type=str, required=True, help='filename or - for stdin')
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--length', type=int, default=500)
args = parser.parse_args()

with open(args.vocab, 'rb') as t_handle:
    tokenizer = pickle.load(t_handle)

rev_word_map = dict(map(reversed, tokenizer.word_index.items()))
model = load_model(args.model)
n_vocab = model.layers[3].get_output_at(0).get_shape().as_list()[-1]
assert n_vocab == len(rev_word_map), "vocab size different than trained vocab"
max_len = model.layers[0].get_output_at(0).get_shape().as_list()[-1] ** 3


def slice_seq(padded_seqs):
    ret = []

    split = np.split
    for i in range(padded_seqs.shape[0]):
        S = split(padded_seqs[i], 8)
        a = []
        for j in range(8):
            a.append(split(S[j], 8))

        ret.append(a)

    return ret


def predict_next(text):
    seqs              = tokenizer.texts_to_sequences([text])
    padded_seqs       = pad_sequences(seqs, maxlen=max_len)
    padded_seqs_split = slice_seq(padded_seqs)

    pred = model.predict([padded_seqs_split], verbose=False)[0]
    pred = pred / args.temperature
    
    # softmax
    pred = np.exp(pred - np.max(pred))
    pred = pred / pred.sum()

    return pred

def search(text, depth):
    pred  = predict_next(text[-max_len:])

    index = np.random.choice(len(pred), 1, p=pred)[0]+1
    text = text + rev_word_map[index]

    if depth > 0:
        biggest = np.argmax(pred)
        top_n = np.flip(np.argpartition(pred, -4)[-4:], 0)


text = args.text
for i in tqdm(range(args.length)):
    pred  = predict_next(text[-max_len:])
    index = np.random.choice(len(pred), 1, p=pred)[0]+1
    text = text + rev_word_map[index]

print(text)
