# SRNN
Author: Zeping Yu
This work is accepted by COLING 2018. The paper could be downloaded at https://arxiv.org/ftp/arxiv/papers/1807/1807.02291.pdf
Sliced Recurrent Neural Network (SRNN).
SRNN is able to get much faster speed than standard RNN by slicing the sequences into many subsequences.
The code is written in keras, using tensorflow backend. We implement the SRNN(8,2) here, and Yelp 2013 dataset is used.
If you have any question, please contact me at zepingyu@foxmail.com.

The pre-trained GloVe word embeddings could be downloaded at:
https://nlp.stanford.edu/projects/glove/

The Yelp 2013, 2014 and 2015 datasets are at:
https://figshare.com/articles/Yelp_2013/6292142
https://figshare.com/articles/Untitled_Item/6292253
https://figshare.com/articles/Yelp_2015/6292334

Yelp_P, Amazon_P and Amazon_F datasets are at: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

## Instructions

```
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt

# try --help on these scripts to see different options
python preprocess.py --csv yelp_2013.csv --output dataset.h5
python train.py --dataset dataset.h5

# view training with tensorboard
tensorboard --logdir logs/

# pass either - for stdin or a filename to text to predict
echo "really really good" | ./predict.py --model save_model/2018-08-24-14-57-21.h5 --text -
```

