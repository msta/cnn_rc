# coding: utf-8
import numpy as np
import re
import math
import word2vec
import argparse

from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Input, merge, Flatten, Reshape
from keras.layers import Convolution1D as Conv1D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling1D
from keras.regularizers import l2

from prep import Preprocessor
from model import get_model

from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument("-d", "--debug", default=False, type=bool)
parser.add_argument("-p", "--with_pos", default=False, type=bool)
parser.add_argument("-bi", "--bidirectional_classes", default=True, type=bool)
args = parser.parse_args()

DEBUG = args.debug
DROPOUT_RATE = 0.5

INCLUDE_POS_EMB = args.with_pos

WORD_EMBEDDING_DIMENSION = 300

if DEBUG:
    EPOCHS = 2
    word_embeddings = {}
    oov_vector = np.zeros(WORD_EMBEDDING_DIMENSION)
else:
    EPOCHS = 100
    word_embeddings = word2vec.load("word_embeddings.bin", encoding='ISO-8859-1')
    oov_vector = np.mean(word_embeddings.vectors[len(word_embeddings.vectors) - 1000:], axis=0)
    #word_embeddings = {}

# classes in the problem
output_dict = {
    "Cause-Effect(e1,e2)" : 0,
    "Cause-Effect(e2,e1)" : 1,
    "Instrument-Agency(e1,e2)" : 2,
    "Instrument-Agency(e2,e1)" : 3,
    "Product-Producer(e1,e2)" : 4,
    "Product-Producer(e2,e1)" : 5,
    "Content-Container(e1,e2)" : 6,
    "Content-Container(e2,e1)" : 7,
    "Entity-Origin(e1,e2)" : 8,
    "Entity-Origin(e2,e1)" : 9,
    "Entity-Destination(e1,e2)" : 10,
    "Entity-Destination(e2,e1)" : 11,
    "Component-Whole(e1,e2)" : 12,
    "Component-Whole(e2,e1)" : 13,
    "Member-Collection(e1,e2)" : 14,
    "Member-Collection(e2,e1)" : 15,
    "Message-Topic(e1,e2)" : 16,
    "Message-Topic(e2,e1)" : 17,
    "Other" : 18
}

def clean_classes(dict):
    clean = set([x.split("(")[0] for x in dict])
    count = 0
    o = {}
    for c in clean:
        o[c] = count
        count += 1
    return o
if not args.bidirectional_classes:
    output_dict = clean_classes(output_dict)
NO_OF_CLASSES = len(output_dict)



def process_train(line):
    # split on quotes and get the raw text
    stripped = re.sub("\d{0,4}", "", line, count=1).strip()
    return stripped[1:len(stripped)-2]

# load dataset
if DEBUG:
    dataset = open("task8/training/TRAIN_FILE_MEDIUM.txt", "r")
else:
    dataset = open("task8/training/TRAIN_FILE.txt", "r")

# input text
X_raw = []

# pred
Y = []

# process dataset
i = 0

for line in dataset:
    if i == 0:
        X_raw.append(process_train(line))
    if i == 1:
        if not args.bidirectional_classes:
            Y.append(output_dict[line.strip().split("(")[0]])
        else:
            Y.append(output_dict[line.strip()])
    if i == 2 or i == 3:
        pass
    i += 1
    if i % 4 == 0:
        i = 0

print "#" * 30
print "DATA SPLIT AND LOADED"
print "#" * 30

# In[300]:

X_full = np.asarray(X_raw)
Y_full = np.asarray(Y) 

prep = Preprocessor(X_full, DEBUG)

X_padded, X_nom_pos1, X_nom_pos2 = prep.preprocess(X_full)
Y = Y_full

word_index = prep.word_idx()

n = prep.n

def debug_print(input, msg):
    print "#" * 30
    print msg
    for line in input:
        print line
    print "#" * 30


if DEBUG:
    print "#" * 30
    print "DEBUG INFO"
    debug_print(X_raw, "Training samples")
    debug_print(X_padded, "Embedding Input")
    debug_print(X_nom_pos, "Nominal positions: ")
    print "-" * 30
    print "Word_index: "
    print word_index
    print "Word idx length: ", len(word_index) 
    print "-" * 30



debug_print([], "SENTENCES SEQUENCED AND NOMINAL POSITIONS CALCULATED")

model = get_model(word_embeddings, 
    word_index, 
    n, 
    WORD_EMBEDDING_DIMENSION,
    INCLUDE_POS_EMB,
    DROPOUT_RATE,
    NO_OF_CLASSES, 
    oov_vector)

X=[X_padded]

if INCLUDE_POS_EMB:
    X.append(X_nom_pos1, X_nom_pos2)

#X = [X_padded, X_nom_pos]
Y_final = to_categorical(Y, nb_classes=NO_OF_CLASSES)

#Y_test = to_categorical(Y_test, nb_classes=NO_OF_CLASSES) 

def train_model_kv(model):
       # print X_padded[train]
        print model.summary()
        #print model.get_config()
        #print model.get_weights();
        model.fit(X, Y_final, nb_epoch=EPOCHS, validation_split=0.1, batch_size=50, shuffle=True)
        #print model.get_weights();
train_model_kv(model)


