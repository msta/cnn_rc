    # coding: utf-8
import numpy as np
import re
import math
import word2vec
import logging
import pickle 
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

from .supersense import SupersenseLookup
from .functions import *
from .semeval_prep import *
from .model import get_model, train_model
from .argparser import build_argparser

########## Parse arguments and setup logging ####################
#################################################################

parser = build_argparser()
args = parser.parse_args()


loglevel = logging.DEBUG if args.debug else logging.INFO

logging.basicConfig(
    level=loglevel,
    format="[%(asctime)s] %(levelname)s:%(message)s")


WINDOW_SIZE = args.filter_size
WINDOW_HEIGHT = args.window_sizes

FOLDS = args.folds 
EPOCHS = args.epochs
DEBUG = args.debug
TEST_FILE = args.test_file
DROPOUT_RATE = args.dropoutrate
INCLUDE_POS_EMB = not args.no_pos
WORD_EMBEDDING_DIMENSION = args.wordembeddingdim
INCLUDE_ATTENTION_ONE = args.attention_one
INCLUDE_ATTENTION_TWO = args.attention_two
POS_EMBEDDING_DIM = args.posembeddingdim
L2_VALUE = args.l2
TRAIN_FILE = args.train_file
EXCLUDE_OTHER = args.exclude_other
INCLUDE_WORDNET = args.wordnet
EMBEDDING = args.embedding
CLIPPING_VALUE = args.clipping

objectives = { "ranking_loss" : ranking_loss,
                "categorical_crossentropy" : "categorical_crossentropy"  }
OBJECTIVE = objectives[args.loss]


no_of_clz = len(output_dict)

NO_OF_CLASSES = no_of_clz - 1 if args.exclude_other else no_of_clz

########## Load embeddings, dataset and prep input ####################
#######################################################################


if EMBEDDING == 'word2vec':
    word_embeddings = word2vec.load("word_embeddings.bin", encoding='ISO-8859-1')
elif EMBEDDING == 'glove':
    word_embeddings = pickle.load(open("glove300b.pkl", "rb"))
    # word_embeddings = {}
    # f = open("deps.words")
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     word_embeddings[word] = coefs
    # f.close()
elif EMBEDDING == 'rand':
    word_embeddings = {}

prep = Preprocessor(clipping_value=CLIPPING_VALUE)

dataset_full, labels_full = prep.read_dataset(TRAIN_FILE, debug=DEBUG,
                                              EXCLUDE_OTHER=EXCLUDE_OTHER)


logging.info("#" * 30)
logging.info("DATA SPLIT AND LOADED")
logging.info("#" * 30)


(word_input, nom_pos_1, nom_pos_2, 
    att_idx, att_list_1, att_list_2,
    Y) = prep.fit_transform(dataset_full, labels_full)


wordnet_sequences = []
ss_lookup = SupersenseLookup()

if INCLUDE_WORDNET:
    tags = open("data/semeval/training/" + TRAIN_FILE + ".tags").read()
    ss_lookup.fit(tags)
    wordnet_sequences = ss_lookup.transform(prep.reverse_sequence(word_input))



word_index = prep.word_idx()
clipping_value = prep.clipping_value


########## Print optional debugging output ############################
#######################################################################

zeros = [x for b in word_input for x in b if x == 0]
logging.debug("Total amt of zeros " + str(len(zeros)))
logging.debug("Avg zeros " + str(len(zeros) / len(word_input)))

#####




debug_print(dataset_full, "Training samples")
debug_print(word_input, "Embedding Input")
debug_print(nom_pos_1, "Nominal positions1: ")
debug_print(nom_pos_2, "Nominal positions2: ")
debug_print(prep.reverse_sequence(word_input), "Reverse")
debug_print_dict(word_index, "Word index")

########## Begin K-Fold validation experiment #########################
#######################################################################


if FOLDS > 0:
    logging.info("Ready for K-Fold Validation...")
    logging.info(str(FOLDS) + " folds...")

    all_results = []
    kf = KFold(n_splits=FOLDS)

    fold = 0

    for train_idx, test_idx in kf.split(word_input):
        logging.info("Beginning fold: " +  str(fold))
        fold += 1
        model = get_model( 
            word_embeddings=word_embeddings,
            word_index=word_index, 
            n=clipping_value,
            word_entity_dictionary=att_idx, 
            POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
            L2_VALUE=L2_VALUE,
            WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
            WINDOW_HEIGHT=WINDOW_HEIGHT,
            INCLUDE_POS_EMB=INCLUDE_POS_EMB,
            WORDNET=INCLUDE_WORDNET,
            WORDNET_LEN=len(ss_lookup.supersense_id),
            INCLUDE_ATTENTION_ONE=INCLUDE_ATTENTION_ONE,
            INCLUDE_ATTENTION_TWO=INCLUDE_ATTENTION_TWO,
            DROPOUT_RATE=DROPOUT_RATE,
            WINDOW_SIZE=WINDOW_SIZE,
            NO_OF_CLASSES=NO_OF_CLASSES,
            optimizer=args.optimizer,
            loss=OBJECTIVE
            )

        logging.info(model.summary())

        X_train = [word_input[train_idx]]
        X_test = [word_input[test_idx]]
        
        Y = np.asarray(Y)

        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        build_input_arrays_folded(X_train, 
                                  X_test, 
                                  INCLUDE_POS_EMB, 
                                  INCLUDE_ATTENTION_ONE,
                                  INCLUDE_WORDNET,
                                  nom_pos_1,
                                  nom_pos_2,
                                  att_list_1,
                                  att_list_2,
                                  wordnet_sequences,
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


    ### Random bug with the final fold ? ### 
    final_f1 = sum([x[2] for x in all_results[:-1] if not np.isnan(x[2]) ]) / (FOLDS-1)

    logging.info( "#" * 30)
    logging.info( "FINAL F1 VALUE: " + str(final_f1))
    logging.info( "#" * 30 )


########## Use all training data for official #########################
########## SemEval F1 scorer                  #########################
#######################################################################


else:
    import keras.backend.tensorflow_backend as K
    with K.tf.device('/gpu:0'):
        logging.info("Beginning test with official F1 scorer ...")

        test_path = "data/semeval/testing/" + TEST_FILE + ".txt"

        test_set = open(test_path)

        data, ids = prep.read_testset(test_set, output_dict)

        with open("test_text_only.txt", "w+") as f:
            for t in data:
                rep = t.replace("<e1>", "").replace("</e1>", "").replace("</e2>", "").replace("<e2>", "")
                f.write(rep + "\n")


        model = get_model( 
                word_embeddings=word_embeddings,
                word_index=word_index, 
                n=clipping_value,
                word_entity_dictionary=att_idx, 
                POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
                L2_VALUE=L2_VALUE,
                WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
                INCLUDE_POS_EMB=INCLUDE_POS_EMB,
                WORDNET=INCLUDE_WORDNET,
                WORDNET_LEN=len(ss_lookup.supersense_id),
                WINDOW_HEIGHT=WINDOW_HEIGHT,
                INCLUDE_ATTENTION_ONE=INCLUDE_ATTENTION_ONE,
                INCLUDE_ATTENTION_TWO=INCLUDE_ATTENTION_TWO,
                DROPOUT_RATE=DROPOUT_RATE,
                WINDOW_SIZE=WINDOW_SIZE,
                NO_OF_CLASSES=NO_OF_CLASSES,
                optimizer=args.optimizer,
                loss=OBJECTIVE
                )

        logging.info(model.summary())

        X_train = [word_input]

        build_input_arrays_test(X_train, INCLUDE_POS_EMB, INCLUDE_ATTENTION_ONE, INCLUDE_WORDNET,
                                att_list_1, att_list_2,
                                nom_pos_1, nom_pos_2,
                                wordnet_sequences)

        Y_train = build_label_representation(Y, OBJECTIVE=OBJECTIVE,
                                    NO_OF_CLASSES=NO_OF_CLASSES,
                                    WINDOW_SIZE=WINDOW_SIZE)

        train_model(model, X_train, Y_train, EPOCHS)
        logging.info("Done training...")

    ########## Prepare test data                  #########################
    #######################################################################


        (X_test, X_test_nom_pos1, X_test_nom_pos2, 
        att_idx, att_test_list_1, att_test_list_2, kept_ids) = prep.transform(data,ids)

        word_idx = prep.word_idx()

        if INCLUDE_WORDNET:
            tags = open("data/semeval/testing/" + TEST_FILE + ".tags").read()
            ss_lookup.fit(tags)
            wordnet_sequences = ss_lookup.transform(prep.reverse_sequence(X_test))




        X = [X_test]


        build_input_arrays_test(X, INCLUDE_POS_EMB, INCLUDE_ATTENTION_ONE, INCLUDE_WORDNET,
                                att_test_list_1, att_test_list_2,
                                X_test_nom_pos1, X_test_nom_pos2,
                                wordnet_sequences)

        failures = [x for x in ids if x not in kept_ids]

        logging.info("Predicting for X...")    
        preds = model.predict(X)
            
        exclude_ratios = [0.20, 0.30, 0.4, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

            
        if args.exclude_other:

            for e_r in exclude_ratios:

                def check_other(pred):
                   return np.argmax(pred) if np.max(pred) > e_r else NO_OF_CLASSES


                preds_final = [check_other(pred) for pred in preds]
                lookup_labels = [reverse_dict[pred] for pred in preds_final]

                with open("data/semeval/test_pred.txt" + str(e_r), "w+") as f:
                    for idx, i in enumerate(kept_ids):
                        f.write(i + "\t" + lookup_labels[idx] + "\n")


        else:
            preds_final = [np.argmax(pred) for pred in preds]

            lookup_labels = [reverse_dict[pred] for pred in preds_final]

            with open("data/semeval/test_pred.txt", "w+") as f:
                for idx, i in enumerate(kept_ids):
                    f.write(i + "\t" + lookup_labels[idx] + "\n")

                for i in failures:
                    f.write(i + "\t" + "Other" + "\n")

logging.info("Experiment done!")
