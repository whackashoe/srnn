# SRNN


Author: Zeping Yu
This work is accepted by COLING 2018. The paper could be downloaded at https://arxiv.org/ftp/arxiv/papers/1807/1807.02291.pdf
Sliced Recurrent Neural Network (SRNN).
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.
If you have any question, please contact me at zepingyu@foxmail.com.


## Text Generation

I modified Zeping's work to predict next characters. Here is a sample generated using:

`./train.py --input datasets/tinyshakespeare/input.txt --vocab datasets/tinyshakespeare/input.txt.pickle`

```
romeo:
i will i have with the duke of the will the 'stands the earth,
the trubutes the rest the death,
i will recompensent with a man the winder and live the nears and zind the death,
when lie the will the court is this death,
where in the respection the worse
that is the prince poise
that xick the capither that is with the britwess with the rest
the -way is ? ready your success
```

Other interesting data sets to try is the [Cornell Movie Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and the [Gutenberg](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)



## Update Instructions

```
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt

# try --help on these scripts to see different options
./generate_tokens.py --input datasets/tinyshakespeare/input.txt
./train.py --input datasets/tinyshakespeare/input.txt --vocab datasets/tinyshakespeare/input.txt.pickle

# view training with tensorboard
tensorboard --logdir logs/

# generate text from trained model
./predict.py --model datasets/tinyshakespeare/input.txt.h5 --vocab datasets/tinyshakespeare/input.txt.pickle
```


