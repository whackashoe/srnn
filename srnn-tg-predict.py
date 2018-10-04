#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd

import argparse
import os
import time
import sys

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, required=True, help='')
parser.add_argument('--model', type=str, required=True, help='')
parser.add_argument('--max_num_words', type=int, default=30000, help='')
parser.add_argument('--text', type=str, required=True, help='filename or - for stdin')
args = parser.parse_args()

raw_text = open(args.input).read().lower()
tokenizer = Tokenizer(num_words=args.max_num_words, char_level=True)
tokenizer.fit_on_texts([text])
vocab = tokenizer.word_index
print(vocab)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

index = numpy.random.multinomial(1, p=softmax(prediction))






text = ""

model = load_model(args.model)
n_vocab = model.layers[3].get_output_at(0).get_shape().as_list()[-1]
slice_width = model.layers[0].get_output_at(0).get_shape().as_list()[-1]
max_len = (slice_width**3)


word_ids = tokenizer.texts_to_sequences([text])
padded_seqs = pad_sequences(word_ids, maxlen=max_len)
padded_seqs_split = seqs_split(padded_seqs, slice_width)

pred = model.predict([padded_seqs_split], verbose=False)[0]
print(pred)
print(sum(star*prob for star, prob in zip([1,2,3,4,5], pred)))
