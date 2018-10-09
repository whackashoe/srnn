#!/usr/bin/env python3

import argparse
import os
import time
from tqdm import tqdm
import pickle
from keras.preprocessing.text import Tokenizer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, required=True, help='')
parser.add_argument('--output', type=str, required=False, default=None, help='')
parser.add_argument('--max-tokens', type=int, default=256, help='')
args = parser.parse_args()

tokenizer = Tokenizer(num_words=args.max_tokens, char_level=True)

raw_text = open(args.input)
lines = raw_text.readlines()
n_lines = len(lines)
for idx, line in enumerate(tqdm(lines)):
    tokenizer.fit_on_texts(line)

output = args.output
if output is None:
    output = args.input + ".pickle"

with open(output, 'wb') as t_handle:
    pickle.dump(tokenizer, t_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} written'.format(args.input + ".pickle"))
