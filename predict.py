#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd

import argparse
import os
import time
import sys


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, required=True, help='')
parser.add_argument('--max_num_words', type=int, default=30000, help='')
parser.add_argument('--max_len', type=int, default=512, help='')
parser.add_argument('--text', type=str, required=True, help='filename or - for stdin')
args = parser.parse_args()

text = ""

if args.text == "-":
    lines = []
    for line in sys.stdin:
        lines.append(line)
    text = ''.join(str(x) for x in lines)
else:
    with open(args.text, 'r') as f:
        text = ''.join(str(x) for x in f.readlines())

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from util import seqs_split

model = load_model(args.model)


tokenizer = Tokenizer(num_words=args.max_num_words)
tokenizer.fit_on_texts([text])
vocab = tokenizer.word_index

word_ids = tokenizer.texts_to_sequences([text])
padded_seqs = pad_sequences(word_ids, maxlen=args.max_len)
padded_seqs_split = seqs_split(padded_seqs)

pred = model.predict([padded_seqs_split], verbose=False)[0]
print(pred)
print(sum(star*prob for star, prob in zip([1,2,3,4,5], pred)))
