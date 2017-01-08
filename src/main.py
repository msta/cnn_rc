# coding: utf-8
import numpy as np
import re
import math
import word2vec
import argparse
import logging


from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Input, merge, Flatten, Reshape
from keras.layers import Convolution1D as Conv1D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import KFold

### My own stuff ###
from semeval import output_dict
from functions import debug_print, fbetascore, clean_classes, process_train
from functions import debug_print_dict, margin_loss
from prep import Preprocessor
from model import get_model



parser = argparse.ArgumentParser(description='CNN')
parser.add_argument("--debug", 
                    action="store_true")
parser.add_argument("--no_pos",  
                    action="store_true",
                    default=False)
parser.add_argument("-f", "--folds", 
                    type=int,
                    default=10)
parser.add_argument("-a", "--attention", 
                    action="store_true",
                    default=False)
parser.add_argument("-e", "--epochs",
                    type=int,
                    default=20)
parser.add_argument("--clipping",
                    type=int,
                    default=18)
parser.add_argument("--merge_classes",  
                    action="store_true")
parser.add_argument("-r", "--rand",  
                    action="store_true")
parser.add_argument("--markup",  
                    action="store_true")
parser.add_argument("-o", "--optimizer",
                    type=str,
                    default='ada',
                    choices=["sgd", "ada"])
parser.add_argument("--windowsize",
                    type=int,
                    default=400)
parser.add_argument("-loss",
                    type=str,
                    default="categorical_crossentropy")




args = parser.parse_args()



if args.loss == 'margin_loss':
    loss = margin_loss
else:
    loss = args.loss


loglevel = logging.DEBUG if args.debug else logging.INFO

logging.basicConfig(
    level=loglevel,
    format="[%(asctime)s] %(levelname)s:%(message)s")


if args.merge_classes:
    logging.info("Cleaning classes")
    output_dict = clean_classes(output_dict)

def read_dataset(dataset, output_dict, merge_classes=False):
    X_raw = []
    Y = []
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

######## Experiment begin ##################

WINDOW_SIZE = args.windowsize
FOLDS = args.folds 
EPOCHS = args.epochs
DEBUG = args.debug
DROPOUT_RATE = 0.5
INCLUDE_POS_EMB = not args.no_pos
WORD_EMBEDDING_DIMENSION = 300
INCLUDE_ATTENTION = args.attention

NO_OF_CLASSES = len(output_dict)


if DEBUG:
    EPOCHS = 2

if not args.rand:
    word_embeddings = word2vec.load("word_embeddings.bin", encoding='ISO-8859-1')
else:
    word_embeddings = {}


# load dataset
if DEBUG:
    dataset = open("task8/training/TRAIN_FILE_SMALL.txt", "r")
else:
    dataset = open("task8/training/TRAIN_FILE.txt", "r")



#############################################################################
############### DATASET READING #############################################
#############################################################################

X_raw, Y = read_dataset(dataset, output_dict, args.merge_classes)

logging.info("#" * 30)
logging.info("DATA SPLIT AND LOADED")
logging.info("#" * 30)

X_full = np.asarray(X_raw)
Y_full = np.asarray(Y) 


prep = Preprocessor(texts=X_full, 
                    Y=Y_full, 
                    debug=DEBUG,
                    clipping_value=args.clipping,
                    markup=args.markup)


(X_padded, X_nom_pos1, X_nom_pos2, 
    att_idx, att_list_1, att_list_2,
    Y) = prep.preprocess(X_full)

### Calculate avg nr of zeros for info

zeros = [x for b in X_padded for x in b if x == 0]
print "Total amt of zeros " , len(zeros)
print "Avg zeros " , len(zeros) / len(X_padded)

#####


word_index = prep.word_idx()
n = prep.n

debug_print(X_raw, "Training samples")
debug_print(X_padded, "Embedding Input")
debug_print(X_nom_pos1, "Nominal positions1: ")
debug_print(X_nom_pos2, "Nominal positions2: ")
debug_print(prep.reverse_sequence(X_padded), "Reverse")
debug_print_dict(word_index, "Word index")

### Beginning K-Fold validation
logging.info("SENTENCES SEQUENCED AND NOMINAL POSITIONS CALCULATED")

all_results = []
kf = KFold(n_splits=FOLDS)

for train_idx, test_idx in kf.split(X_padded):

    model = get_model( 
        word_embeddings=word_embeddings,
        word_index=word_index, 
        n=n,
        word_entity_dictionary=att_idx, 
        WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
        INCLUDE_POS_EMB=INCLUDE_POS_EMB,
        INCLUDE_ATTENTION=INCLUDE_ATTENTION,
        DROPOUT_RATE=DROPOUT_RATE,
        WINDOW_SIZE=WINDOW_SIZE,
        NO_OF_CLASSES=NO_OF_CLASSES,
        optimizer=args.optimizer,
        loss=loss
        )


    X_train = [X_padded[train_idx]]
    X_test = [X_padded[test_idx]]
    
    if INCLUDE_POS_EMB:
        X_train.append(X_nom_pos1[train_idx])
        X_train.append(X_nom_pos2[train_idx])
        X_test.append(X_nom_pos1[test_idx])
        X_test.append(X_nom_pos2[test_idx])

    if INCLUDE_ATTENTION:
        X_train.append(att_list_1[train_idx])
        X_train.append(att_list_2[train_idx])

        X_test.append(att_list_1[test_idx])
        X_test.append(att_list_2[test_idx])


    #X = [X_padded, X_nom_pos]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    def transform_to_embedding(label, nb_classes, dimensions):
        m = np.zeros((nb_classes, dimensions))
        m[label,:] = 1
        return m


    if loss in ['squared_hinge', 'cosine_proximity']:
        embeddings = np.random.rand(NO_OF_CLASSES, WINDOW_SIZE)
        Y_train = np.asarray([embeddings[y] for y in Y_train])
        Y_test = np.asarray([embeddings[y] for y in Y_test])

    

    elif loss == 'categorical_crossentropy' or loss == margin_loss:
    
        Y_train = to_categorical(Y_train, nb_classes=NO_OF_CLASSES)
        Y_test = to_categorical(Y_test, nb_classes=NO_OF_CLASSES)
    # elif loss == margin_loss:
    #     Y_train = [transform_to_embedding(label, NO_OF_CLASSES, WINDOW_SIZE) for label in Y_train]
    #     Y_test = [transform_to_embedding(label, NO_OF_CLASSES, WINDOW_SIZE) for label in Y_test]



    def train_model_kv(model):
            logging.info(model.summary())
            history = model.fit(X_train, 
                Y_train, 
                nb_epoch=EPOCHS, 
                batch_size=50, 
                shuffle=True)
            
    train_model_kv(model)

    logging.info( "#" * 30)
    logging.info( "EVALUATING MODEL")

    result = model.evaluate(X_test, Y_test)
    all_results.append(result)
    logging.info( result)
    logging.info( "EVALUATING DONE")
    logging.info( "#" * 30)


final_f1 = sum([x[2] for x in all_results if not np.isnan(x[2]) ]) / FOLDS

logging.info( "#" * 30)
logging.info( "FINAL F1 VALUE: " , final_f1)
logging.info( "#" * 30 )


