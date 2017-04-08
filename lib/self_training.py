    # coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


import numpy as np
import re
import math
import word2vec
import logging
import pickle 

from collections import defaultdict
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
from sklearn.metrics import f1_score

from .supersense import SupersenseLookup
from .functions import *
from .model import get_model, train_model
from .pretrained_model import get_pretrained_model

from .argparser import build_argparser

def main(args):

    
    ########## Parse arguments and setup logging ####################
    #################################################################
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
    POS_EMBEDDING_DIM = args.posembeddingdim
    L2_VALUE = args.l2
    TRAIN_FILE = args.train_file
    EXCLUDE_OTHER = args.exclude_other
    EMBEDDING = args.embedding
    CLIPPING_VALUE = args.clipping
    DATASET = args.dataset

    THROTTLING = 300
    THRESHOLD = 0.85
    objectives = { "categorical_crossentropy" : "categorical_crossentropy"  }
    OBJECTIVE = objectives[args.loss]

    ########## Load embeddings, dataset and prep input ####################
    #######################################################################
    if EMBEDDING == 'word2vec':
        logging.info("Loading word2vec embeddings")
        word_embeddings = word2vec.load("embeddings/word_embeddings.bin", encoding='ISO-8859-1')
        logging.info("Done!")
        # fit_str = ""
        # for word in word_embeddings.vocab:
            # fit_str += word + " "
    elif EMBEDDING == 'glove':
        word_embeddings = pickle.load(open("embeddings/glove300b.pkl", "rb"))

    elif EMBEDDING == 'rand':
        word_embeddings = {}

    if DATASET == 'semeval':
        from .semeval.prep import Preprocessor, output_dict, reverse_dict
        prep = Preprocessor(clipping_value=CLIPPING_VALUE)
        dataset_full, labels_full = prep.read_dataset(TRAIN_FILE, debug=DEBUG,
                                                      EXCLUDE_OTHER=EXCLUDE_OTHER)
        dataset_aug, _, _ = prep.read_semiset("examples_wiki", debug=DEBUG,
                                                        EXCLUDE_OTHER=EXCLUDE_OTHER)

    #### Compute number of output classes #################################
    no_of_clz = len(output_dict)
    NO_OF_CLASSES = no_of_clz - 1 if args.exclude_other else no_of_clz

    logging.info("#" * 30)
    logging.info("DATA SPLIT AND LOADED")
    logging.info("#" * 30)

    ########

    (word_input, nom_pos_1, nom_pos_2, 
        e_pairs, _, _,
        Y) = prep.fit_transform(dataset_full, labels_full)
    X = [word_input, nom_pos_1, nom_pos_2]


    logging.info("Ready for K-Fold Validation...")
    logging.info(str(FOLDS) + " folds...")

    all_results = []
    kf = KFold(n_splits=FOLDS, shuffle=True)

    
    Y = build_label_representation(Y, OBJECTIVE=OBJECTIVE,
                                NO_OF_CLASSES=NO_OF_CLASSES)

    ids = [i for i in range(0, len(dataset_aug))]
    (word_input_aug, nom_pos_aug_1, nom_pos_aug_2, 
    aug_e_pairs, aug_texts, _,
    kept_ids) = prep.fit_transform(dataset_aug, ids)

    X_aug = [word_input_aug, nom_pos_aug_1, nom_pos_aug_2]

    word_index = prep.word_idx()
    clipping_value = prep.clipping_value

    all_results = []

    X_aug_backup = X_aug

    for train_val_idx, test_idx in list(kf.split(X[0]))[:1]:
        
        # ## 10% val split
        train_idx, val_idx = train_test_split(train_val_idx, test_size = 0.1)

        ## KF split returns indices, while testsplit returns values

        result = 0.0

        fold_e_pairs = e_pairs[train_idx]
        X_fold = [xx[train_idx] for xx in X]
        X_val = [xx[val_idx] for xx in X]
        X_test = [xx[test_idx] for xx in X]


        Y_fold = Y[train_idx]
        Y_test = Y[test_idx]
        Y_val = Y[val_idx]

        ## FOr throttling and balancing
        clz_count = defaultdict(int)
        fold_clz_count = defaultdict(int)
        
        for yf in [np.argmax(y_) for y_ in Y_fold]:
            # merge semeval classes
            # if yf % 2 == 1:
                # yf -= 1
            fold_clz_count[yf] += 1


        fold_size = Y_fold.shape[0]
        distrib_clz = { k : float(v) / float(fold_size)  for k, v in fold_clz_count.items() }

        distrib_100 = defaultdict(int, { k : math.ceil( v*THROTTLING ) for k,v in distrib_clz.items() })

        while True:              
            model = get_model( 
                word_embeddings=word_embeddings,
                word_index=word_index, 
                n=clipping_value,
                POS_EMBEDDING_DIM=POS_EMBEDDING_DIM,
                L2_VALUE=L2_VALUE,
                WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIMENSION,
                WINDOW_HEIGHT=WINDOW_HEIGHT,
                DROPOUT_RATE=DROPOUT_RATE,
                WINDOW_SIZE=WINDOW_SIZE,
                NO_OF_CLASSES=NO_OF_CLASSES,
                optimizer=args.optimizer,
                loss=OBJECTIVE)

            history = train_model(model, X_fold, Y_fold, EPOCHS, early_stopping=True, val_data=(X_val, Y_val))
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
                loss=OBJECTIVE)

            best_epoch = np.argmin(np.asarray(history.history['val_loss'])) + 1
            logging.info("Training with " + str(best_epoch) + " epochs by early stopping")
            history = train_model(model, X_fold, Y_fold, best_epoch, early_stopping=False)
            #history = train_model(model, X_fold, Y_fold, EPOCHS, early_stopping=False)

            
            # preds = []
            #for i in range(0, len(X_aug[0])):
            #preds.append(model.predict([x[i:i+1] for x in X_aug]))

            logging.info("Classifying unlabeled data...")    
            preds = model.predict(X_aug)
            

            logging.info("Selecting samples to include in training set...")

            #kept_ids = set(kept_ids)
            best_clz = [np.argmax(p) for p in preds]
            best_prob = [np.max(p) for p in preds]

            correct_preds = 0

            samples_to_add = []
            clz_to_add = []

            ###### Selection of new samples

            min_threshold = 0.8
            # Balanced selection of equal amounts of samples

            for current_clz in range(0, 18, 2):

                brother_clz = current_clz + 1

                # Get all samples of specific class
                clz_preds = np.asarray([idx for idx, clz in enumerate(best_clz) if (clz == current_clz or clz == brother_clz) 
                    and aug_e_pairs[idx] in fold_e_pairs and best_prob[idx] > THRESHOLD])
                if clz_preds.any():

                    no_to_get = distrib_100[current_clz] + distrib_100[brother_clz]

                    clz_probs = np.asarray(best_prob)[clz_preds]

                    indices = np.asarray(list(range(0, len(clz_preds))))

                    best_probs = np.random.choice(indices, size=no_to_get, replace=False)

                    #best_probs = clz_probs.argsort()[-no_to_get:][::-1]

                    samples_to_add.extend(list(clz_preds[best_probs]))
                    clz_to_add.extend(list(np.asarray(best_clz)[clz_preds[best_probs]]))

                    avg_conf = clz_probs[best_probs].mean()
                    logging.info("Selecting %i samples for training. Best average confidence for class %i is %f", len(best_probs), current_clz, avg_conf)
                else:
                    logging.info("No samples found for class %i", current_clz)
            

            # for sentence_id, sentence in enumerate(X_aug):
            #     ## sentence_id == idx in X_aug
            #     # If not failed, advanced sentence_id
            #     if best_prob[sentence_id] > test_threshold[best_clz[sentence_id]] and aug_e_pairs[sentence_id] in fold_e_pairs:
            #         correct_preds += 1
            #         samples_to_add.append(sentence_id)
            #         clz_to_add.append(best_clz[sentence_id])

            logging.info("Found samples to add. " + str(len(samples_to_add)) + " new sentences")

            logging.info("Evaluating on fold for intermediate result")
            Y_pred = model.predict(X_test)
            
            new_result = f1_score([np.argmax(x) for x in Y_pred],[np.argmax(y) for y in Y_test], average='micro')
            
            logging.info(new_result)
            logging.info("#" * 30)

            if new_result < result:
                logging.info("No improvement in score, moving to next fold")
                break

            result = new_result

            clz_to_add = build_label_representation(np.asarray(clz_to_add), OBJECTIVE=OBJECTIVE,
                                NO_OF_CLASSES=NO_OF_CLASSES,
                                WINDOW_SIZE=WINDOW_SIZE)

            X_fold = [np.append(X_fold[k], X_aug[k][samples_to_add], axis=0) for k in range(0, len(X_fold))]

            ### deleting samples from X_aug
            X_aug = [np.delete(x, samples_to_add, axis=0) for x in X_aug] 
            #kept_ids = np.delete(kept_ids, samples_to_add, axis=0)
            Y_fold = np.append(Y_fold, clz_to_add, axis=0)

            logging.info("Creation done, beginning next iteration")

           
    final_f1 = sum(all_results) 
    #final_loss = sum([x[0] for x in all_results  ]) / fold 
    logging.info( "#" * 30)
    logging.info( "FINAL F1 VALUE: " + str(final_f1))
    logging.info( "#" * 30 )

    logging.info("Experiment done!")

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    main(args)