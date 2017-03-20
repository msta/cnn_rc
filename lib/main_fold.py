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

import keras.backend.tensorflow_backend as K
    
from keras.regularizers import l2
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import KFold, train_test_split
from .supersense import SupersenseLookup
from .functions import *
from .model import get_model, train_model
from .argparser import build_argparser


def main(args):
    ########## Parse arguments and setup logging ####################
    #################################################################
    with K.tf.device('/gpu:0'):

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
        DATASET = args.dataset

        objectives = {
                        "categorical_crossentropy" : "categorical_crossentropy"  }
        OBJECTIVE = objectives[args.loss]




        ########## Load embeddings, dataset and prep input ####################
        #######################################################################
        if EMBEDDING == 'word2vec':
            word_embeddings = word2vec.load("word_embeddings.bin", encoding='ISO-8859-1')
        elif EMBEDDING == 'glove':
            word_embeddings = pickle.load(open("glove300b.pkl", "rb"))
        elif EMBEDDING == 'rand':
            word_embeddings = {}

        if DATASET == 'semeval':
            from .semeval.prep import Preprocessor, output_dict
            prep = Preprocessor(clipping_value=CLIPPING_VALUE)
            dataset_full, labels_full = prep.read_dataset(TRAIN_FILE, debug=DEBUG,
                                                      EXCLUDE_OTHER=EXCLUDE_OTHER)
        elif DATASET == 'ace2005':
            from .ace2005.prep import AcePrep
            prep = AcePrep(clipping_value=CLIPPING_VALUE)
            dataset_full, labels_full = prep.read_dataset(TRAIN_FILE, debug=DEBUG)
            output_dict = [0,1,2,3,4,5,6]


        #### Compute number of output classes #################################

        no_of_clz = len(output_dict)

        NO_OF_CLASSES = no_of_clz - 1 if args.exclude_other else no_of_clz


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

        X = [word_input]

        build_input_arrays_folded(X,
                                  INCLUDE_POS_EMB, 
                                  INCLUDE_ATTENTION_ONE,
                                  INCLUDE_WORDNET,
                                  nom_pos_1,
                                  nom_pos_2,
                                  att_list_1,
                                  att_list_2,
                                  wordnet_sequences)


        Y = build_label_representation(Y_train=Y,
                                                OBJECTIVE=OBJECTIVE,
                                                NO_OF_CLASSES=NO_OF_CLASSES,
                                                WINDOW_SIZE=WINDOW_SIZE)

        x_idx = np.asarray(list(range(0, len(X[0]))))

        # training_idx, val_idx = train_test_split(x_idx, test_size=0.1)

        # X_train = [xx[training_idx] for xx in X]
        # X_val = [xx[val_idx] for xx in X]

        # Y_train = Y[training_idx]
        # Y_val = Y[val_idx]
        X_train = X
        Y_train = Y

        ########## Begin K-Fold validation experiment #########################
        #######################################################################

        logging.info("Ready for K-Fold Validation...")
        logging.info(str(FOLDS) + " folds...")

        all_results = []
        kf = KFold(n_splits=FOLDS, shuffle=True)

        fold = 0

        for train_idx, test_idx in kf.split(X_train[0]):
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

            #logging.info(model.summary())

            X_fold = [xx[train_idx] for xx in X_train]
            X_test = [xx[test_idx] for xx in X_train]
            
            #X_fold = [xx[:int(split*len(xx))] for xx in X_fold]

            Y_fold = Y_train[train_idx]

            #Y_fold = Y_fold[:int(split*len(Y_fold))]
            Y_test = Y_train[test_idx]


            train_model(model, X_fold, Y_fold, EPOCHS, validation_split=0.2)

            logging.info("#" * 30)
            logging.info("EVALUATING MODEL")


            result = model.evaluate(X_test, Y_test)
            # from sklearn.metrics import f1_score

            # Y_pred = model.predict(X_test)

            #result = f1_score([np.argmax(x) for x in Y_pred],[np.argmax(y) for y in Y_test], average='macro')

            all_results.append(result)
            logging.info(result)
            logging.info("EVALUATING DONE")
            logging.info("#" * 30)


        ### Random bug with the final fold ? ### 
        final_f1 = sum([x[1] for x in all_results  ]) / (FOLDS)
        final_loss = sum([x[0] for x in all_results  ]) / (FOLDS)
        logging.info( "#" * 30)
        logging.info( "FINAL F1 VALUE: " + str(final_f1))
        logging.info( "#" * 30 )
        logging.info("AVERAGE LOSS :" + str(final_loss))

    logging.info("Experiment done!")

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    main(args)