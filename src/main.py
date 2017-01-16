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


from .dataset import load_dataset, read_dataset, read_testset
from .semeval import output_dict, reverse_dict
from .functions import *
from .prep import Preprocessor
from .model import get_model, train_model


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument("--debug", 
                    action="store_true")
parser.add_argument("--no_pos",  
                    action="store_true",
                    default=False)
parser.add_argument("-f", "--folds", 
                    type=int,
                    default=10)
parser.add_argument("-a1", "--attention_one", 
                    action="store_true",
                    default=False)
parser.add_argument("-a2", "--attention_two", 
                    action="store_true",
                    default=False)
parser.add_argument("-e", "--epochs",
                    type=int,
                    default=10)
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
parser.add_argument("--wordembeddingdim",
                    type=int,
                    default=300)
parser.add_argument("--posembeddingdim",
                    type=int,
                    default=50)
parser.add_argument("-loss",
                    type=str,
                    default="categorical_crossentropy")
parser.add_argument("-dropoutrate",
                    type=float,
                    default=0.5)


args = parser.parse_args()


loglevel = logging.DEBUG if args.debug else logging.INFO

logging.basicConfig(
    level=loglevel,
    format="[%(asctime)s] %(levelname)s:%(message)s")


if args.merge_classes:
    logging.info("Cleaning classes")
    output_dict = clean_classes(output_dict)



######## Experiment begin ##################

WINDOW_SIZE = args.windowsize
FOLDS = args.folds 
EPOCHS = args.epochs
DEBUG = args.debug
DROPOUT_RATE = args.dropoutrate
INCLUDE_POS_EMB = not args.no_pos
WORD_EMBEDDING_DIMENSION = args.wordembeddingdim
INCLUDE_ATTENTION_ONE = args.attention_one
INCLUDE_ATTENTION_TWO = args.attention_two
POS_EMBEDDING_DIM = args.posembeddingdim

CLIPPING_VALUE = args.clipping
OBJECTIVE = args.loss

NO_OF_CLASSES = len(output_dict)


if not args.rand:
    word_embeddings = word2vec.load("word_embeddings.bin", encoding='ISO-8859-1')
else:
    word_embeddings = {}

dataset = load_dataset(DEBUG)
X_full, Y_full = read_dataset(dataset, output_dict, args.merge_classes)


logging.info("#" * 30)
logging.info("DATA SPLIT AND LOADED")
logging.info("#" * 30)


prep = Preprocessor(clipping_value=CLIPPING_VALUE,
                    markup=args.markup)


(X_padded, X_nom_pos1, X_nom_pos2, 
    att_idx, att_list_1, att_list_2,
    Y) = prep.fit_transform(X_full, Y_full)


### Calculate avg nr of zeros for info

zeros = [x for b in X_padded for x in b if x == 0]
logging.debug("Total amt of zeros " + str(len(zeros)))
logging.debug("Avg zeros " + str(len(zeros) / len(X_padded)))

#####

word_index = prep.word_idx()
n = prep.n



debug_print(X_full, "Training samples")
debug_print(X_padded, "Embedding Input")
debug_print(X_nom_pos1, "Nominal positions1: ")
debug_print(X_nom_pos2, "Nominal positions2: ")
debug_print(prep.reverse_sequence(X_padded), "Reverse")
debug_print_dict(word_index, "Word index")


if FOLDS > 0:
    logging.info("Ready for K-Fold Validation...")
    logging.info(str(FOLDS) + " folds...")

    all_results = []
    kf = KFold(n_splits=FOLDS)

    for train_idx, test_idx in kf.split(X_padded):

        model = get_model( 
            word_embeddings=word_embeddings,
            word_index=word_index, 
            n=n,
            word_entity_dictionary=att_idx, 
            POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
            WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
            INCLUDE_POS_EMB=INCLUDE_POS_EMB,
            INCLUDE_ATTENTION_ONE=INCLUDE_ATTENTION_ONE,
            INCLUDE_ATTENTION_TWO=INCLUDE_ATTENTION_TWO,
            DROPOUT_RATE=DROPOUT_RATE,
            WINDOW_SIZE=WINDOW_SIZE,
            NO_OF_CLASSES=NO_OF_CLASSES,
            optimizer=args.optimizer,
            loss=OBJECTIVE
            )


        X_train = [X_padded[train_idx]]
        X_test = [X_padded[test_idx]]
            
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        build_input_arrays_folded(X_train, 
                                  X_test, 
                                  INCLUDE_POS_EMB, 
                                  INCLUDE_ATTENTION_ONE,
                                  X_nom_pos1,
                                  X_nom_pos2,
                                  att_list_1,
                                  att_list_2,
                                  train_idx,
                                  test_idx)


        Y_train, Y_test = build_label_representation(Y_train=Y_train,
                                                    Y_test=Y_test,
                                                    OBJECTIVE=OBJECTIVE,
                                                    NO_OF_CLASSES=NO_OF_CLASSES,
                                                    WINDOW_SIZE=WINDOW_SIZE)

        train_model(model, X_train, Y_train, EPOCHS)

        logging.info("#" * 30)
        logging.info("EVALUATING MODEL")

        result = model.evaluate(X_test, Y_test)
        all_results.append(result)
        logging.info(result)
        logging.info("EVALUATING DONE")
        logging.info("#" * 30)

    final_f1 = sum([x[2] for x in all_results if not np.isnan(x[2]) ]) / FOLDS

    logging.info( "#" * 30)
    logging.info( "FINAL F1 VALUE: " + str(final_f1))
    logging.info( "#" * 30 )

else:
    logging.info("Beginning test with official F1 scorer ...")

    test_set = open("task8/SemEval2010_task8_testing_keys/TEST_FILE_CLEAN.TXT")

    data, ids = read_testset(test_set, output_dict)

    model = get_model( 
            word_embeddings=word_embeddings,
            word_index=word_index, 
            n=n,
            word_entity_dictionary=att_idx, 
            POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
            WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
            INCLUDE_POS_EMB=INCLUDE_POS_EMB,
            INCLUDE_ATTENTION_ONE=INCLUDE_ATTENTION_ONE,
            INCLUDE_ATTENTION_TWO=INCLUDE_ATTENTION_TWO,
            DROPOUT_RATE=DROPOUT_RATE,
            WINDOW_SIZE=WINDOW_SIZE,
            NO_OF_CLASSES=NO_OF_CLASSES,
            optimizer=args.optimizer,
            loss=OBJECTIVE
            )

    X_train = [X_padded]

    build_input_arrays_test(X_train, INCLUDE_POS_EMB, INCLUDE_ATTENTION_ONE,
                            att_list_1, att_list_2,
                            X_nom_pos1, X_nom_pos2)

    Y_train = build_label_representation(Y, OBJECTIVE=OBJECTIVE,
                                NO_OF_CLASSES=NO_OF_CLASSES,
                                WINDOW_SIZE=WINDOW_SIZE)

    logging.info("Done training...")
    train_model(model, X_train, Y_train, EPOCHS)


    (X_test, X_test_nom_pos1, X_test_nom_pos2, 
    att_test_list_1, att_test_list_2, id_test) = prep.transform(data, ids)

    word_idx = prep.word_idx()


    X = [X_test]


    build_input_arrays_test(X, INCLUDE_POS_EMB, INCLUDE_ATTENTION_ONE,
                            att_test_list_1, att_test_list_2,
                            X_test_nom_pos1, X_test_nom_pos2)

    failures = [x for x in ids if x not in id_test]

    logging.info("Predicting for X...")    
    preds = model.predict(X)
        
    preds = [np.argmax(pred) for pred in preds]

    lookup_labels = [reverse_dict[pred] for pred in preds]


    with open("task8/test_pred.txt", "w+") as f:
        for idx, i in enumerate(id_test):
            f.write(i + "\t" + lookup_labels[idx] + "\n")

        for i in failures:
            f.write(i + "\t" + "Other" + "\n")


