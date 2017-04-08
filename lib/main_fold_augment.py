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
    with K.tf.device('/gpu:2'):

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

            dataset_aug, labels_aug = prep.read_dataset("wiki_examples_done85", debug=DEBUG,
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
            _, _, _,
            Y) = prep.fit_transform(dataset_full, labels_full)

        (word_input_aug, nom_pos_aug_1, nom_pos_aug_2, 
            _, _, _,
            Y_aug) = prep.fit_transform(dataset_aug, labels_aug)


        word_index = prep.word_idx()
        clipping_value = prep.clipping_value

        #X = [np.append(word_input, word_input_aug, axis=0)]
        X = [word_input]

        #nom_pos_1 = np.append(nom_pos_1, nom_pos_aug_1, axis=0)
        #nom_pos_2 = np.append(nom_pos_2, nom_pos_aug_2, axis=0)

        build_input_arrays_test(X, [], [], 
                                  nom_pos_1,
                                  nom_pos_2, [])

        #Y = np.append(Y, Y_aug)

        Y = build_label_representation(Y_train=Y,
                                                OBJECTIVE=OBJECTIVE,
                                                NO_OF_CLASSES=NO_OF_CLASSES,
                                                WINDOW_SIZE=WINDOW_SIZE)

        Y_aug = build_label_representation(Y_train=Y_aug,
                                                OBJECTIVE=OBJECTIVE,
                                                NO_OF_CLASSES=NO_OF_CLASSES,
                                                WINDOW_SIZE=WINDOW_SIZE)


        #x_idx = np.asarray(list(range(0, len(X[0]))))

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
        kf = KFold(n_splits=FOLDS)

        fold = 0

        for train_idx, test_idx in list(kf.split(word_input))[:1]:

            import ipdb
            ipdb.sset_trace()


            logging.info("Beginning fold: " +  str(fold))
            fold += 1
            #logging.info(model.summary())

            X_fold = [xx[train_idx] for xx in X_train]
            X_test = [xx[test_idx] for xx in X_train]
        
            X_fold[0] = np.append(X_fold[0], word_input_aug, axis=0) 
            X_fold[1] = np.append(X_fold[1], nom_pos_aug_1, axis=0) 
            X_fold[2] = np.append(X_fold[2], nom_pos_aug_2, axis=0) 

            

            Y_fold = Y_train[train_idx]
            
            Y_fold = np.append(Y_fold, Y_aug, axis=0)
            Y_test = Y_train[test_idx]

            model = get_model( 
                word_embeddings=word_embeddings,
                word_index=word_index, 
                n=clipping_value,
                POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
                WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
                WINDOW_HEIGHT=WINDOW_HEIGHT,
                DROPOUT_RATE=DROPOUT_RATE,
                WINDOW_SIZE=WINDOW_SIZE,
                NO_OF_CLASSES=NO_OF_CLASSES,
                optimizer=args.optimizer,
                loss=OBJECTIVE
                )

            logging.info(model.summary())

            result = train_model(model, X_fold, Y_fold, EPOCHS, early_stopping=True)
            model = get_model( 
                word_embeddings=word_embeddings,
                word_index=word_index, 
                n=clipping_value,
                POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
                WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
                WINDOW_HEIGHT=WINDOW_HEIGHT,
                DROPOUT_RATE=DROPOUT_RATE,
                WINDOW_SIZE=WINDOW_SIZE,
                NO_OF_CLASSES=NO_OF_CLASSES,
                optimizer=args.optimizer,
                loss=OBJECTIVE
                )

            best_epoch = np.argmin(np.asarray(result.history['val_loss']))

            logging.info("Training with " + str(best_epoch) + " epochs by early stopping")
            
            result = train_model(model, X_fold, Y_fold, best_epoch+1, early_stopping=False)

            logging.info("#" * 30)
            logging.info("EVALUATING MODEL")

            #result = model.evaluate(X_test, Y_test)
            from sklearn.metrics import f1_score

            Y_pred = model.predict(X_test)

            result = f1_score([np.argmax(x) for x in Y_pred],[np.argmax(y) for y in Y_test], average='micro')

            all_results.append(result)
            logging.info(result)
            logging.info("EVALUATING DONE")
            logging.info("#" * 30)


        ### Random bug with the final fold ? ### 
        final_f1 = sum(all_results) / fold
        #final_loss = sum([x[0] for x in all_results  ]) / fold 
        logging.info( "#" * 30)
        logging.info( "FINAL F1 VALUE: " + str(final_f1))
        logging.info( "#" * 30 )
        #logging.info("AVERAGE LOSS :" + str(final_loss))

        logging.info("Experiment done!")

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    main(args)