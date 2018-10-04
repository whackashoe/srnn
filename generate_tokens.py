#!/usr/bin/env python3

import argparse
import os
import time

from console_progressbar import ProgressBar
import pickle


from keras.preprocessing.text import Tokenizer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, required=True, help='')
parser.add_argument('--output', type=str, required=False, default=None, help='')
parser.add_argument('--max_num_words', type=int, default=256, help='')
args = parser.parse_args()

raw_text = open(args.input).read().lower()

tokenizer = Tokenizer(num_words=args.max_num_words, char_level=True)
tokenizer.fit_on_texts(raw_text)

output = args.output
if output is None:
    output = args.input + ".pickle"

with open(output, 'wb') as t_handle:
    pickle.dump(tokenizer, t_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} written'.format(args.input + ".pickle"))
