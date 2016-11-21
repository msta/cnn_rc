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
from keras.callbacks import TensorBoard
from keras import backend as K

from sklearn.model_selection import KFold

from prep import Preprocessor
from model import get_model

from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical

def debug_print(input, msg):
    print "#" * 30
    print msg
    for line in input:
        print line
    print "#" * 30

def fbetascore(y_true, y_pred, beta=1):
    from keras import backend as K
    '''Compute F score, the weighted harmonic mean of precision and recall.
    
    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0
    
    # How many selected items are relevant?
    precision = c1 / c2
    
    # How many relevant items are selected?
    recall = c1 / c3
    
    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score



def clean_classes(dict):
    clean = set([x.split("(")[0] for x in dict])
    count = 0
    o = {}
    for c in clean:
        o[c] = count
        count += 1
    return o


def process_train(line):
    # split on quotes and get the raw text
    stripped = re.sub("\d{0,4}", "", line, count=1).strip()
    return stripped[1:len(stripped)-2]


def read_dataset(dataset, output_dict, merge_classes=False):

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
            if args.merge_classes:
                Y.append(output_dict[line.strip().split("(")[0]])
            else:
                Y.append(output_dict[line.strip()])
        if i == 2 or i == 3:
            pass
        i += 1
        if i % 4 == 0:
            i = 0
    return X_raw, Y


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


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument("--debug", action="store_true")
parser.add_argument("--with_pos",  action="store_true")
parser.add_argument("--merge_classes",  action="store_true")
parser.add_argument("--rand",  action="store_true")
parser.add_argument("--clipping",  action="store_true")
parser.add_argument("--markup",  action="store_true")
args = parser.parse_args()

if args.merge_classes:
    print "Cleaning classes"
    output_dict = clean_classes(output_dict)


######## Experiment begin ##################

FOLDS = 10
EPOCHS = 20
DEBUG = args.debug
DROPOUT_RATE = 0.5
INCLUDE_POS_EMB = args.with_pos
WORD_EMBEDDING_DIMENSION = 300

word_embeddings = {}


NO_OF_CLASSES = len(output_dict)


if DEBUG:
    EPOCHS = 2
else:
    if not args.rand:
        word_embeddings = word2vec.load("word_embeddings.bin", encoding='ISO-8859-1')

# load dataset
if DEBUG:
    dataset = open("task8/training/TRAIN_FILE_SMALL.txt", "r")
else:
    dataset = open("task8/training/TRAIN_FILE.txt", "r")


X_raw, Y = read_dataset(dataset, output_dict, args.merge_classes)


print "#" * 30
print "DATA SPLIT AND LOADED"
print "#" * 30

# In[300]:

X_full = np.asarray(X_raw)
Y_full = np.asarray(Y) 


prep = Preprocessor(X_full, 
    Y_full, 
    DEBUG, 
    args.rand, 
    args.clipping,
    args.markup)

X_padded, X_nom_pos1, X_nom_pos2, Y = prep.preprocess(X_full)

### Calculate avg nr of zeros for info

zeros = [x for b in X_padded for x in b if x == 0]
print "Total amt of zeros " , len(zeros)
print "Avg zeros " , len(zeros) / len(X_padded)

#####


word_index = prep.word_idx()
n = prep.n



if DEBUG:
    print "#" * 30
    print "DEBUG INFO"
    debug_print(X_raw, "Training samples")
    debug_print(X_padded, "Embedding Input")
    debug_print(X_nom_pos1, "Nominal positions1: ")
    debug_print(X_nom_pos2, "Nominal positions2: ")
    debug_print(prep.reverse_sequence(X_padded), "Reverse")
    print "-" * 30
    print "Word_index: "
    print word_index
    print "Word idx length: ", len(word_index) 
    print "-" * 30

### Beginning K-Fold validation
debug_print([], "SENTENCES SEQUENCED AND NOMINAL POSITIONS CALCULATED")

all_results = []
kf = KFold(n_splits=FOLDS)

for train_idx, test_idx in kf.split(X_padded):

    model = get_model( 
        word_embeddings,
        word_index, 
        n, 
        WORD_EMBEDDING_DIMENSION,
        INCLUDE_POS_EMB,
        DROPOUT_RATE,
        NO_OF_CLASSES
        )


    X_train = [X_padded[train_idx]]
    X_test = [X_padded[test_idx]]
    if INCLUDE_POS_EMB:
        X_train.append(X_nom_pos1[train_idx])
        X_train.append(X_nom_pos2[train_idx])
        X_test.append(X_nom_pos1[test_idx])
        X_test.append(X_nom_pos2[test_idx])


    #X = [X_padded, X_nom_pos]
    Y_train = to_categorical(Y[train_idx], nb_classes=NO_OF_CLASSES)
    Y_test = to_categorical(Y[test_idx], nb_classes=NO_OF_CLASSES)


    #Y_test = to_categorical(Y_test, nb_classes=NO_OF_CLASSES) 

    def train_model_kv(model):
           # print X_padded[train]
            #print model.summary()
            #print model.get_config()
            #print model.get_weights();
            model.fit(X_train, Y_train, nb_epoch=EPOCHS, 
                batch_size=50, 
                shuffle=True)
            #print model.get_weights();
    train_model_kv(model)

    print "#" * 30
    print "EVALUATING MODEL"

    result = model.evaluate(X_test, Y_test)
    all_results.append(result)
    print result
    print "EVALUATING DONE"
    print "#" * 30


final_f1 = sum([x[2] for x in all_results if not np.isnan(x[2]) ]) / FOLDS

print "#" * 30
print "FINAL F1 VALUE: " , final_f1
print "#" * 30 


# weights = model.get_weights()
# print len(weights)
# print "#" * 30
# print model.get_weights()[10].shape
# print model.get_weights()[10]
# print "#" * 30
# print model.get_weights()[9].shape
# print model.get_weights()[9]




