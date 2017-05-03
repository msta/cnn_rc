# coding: utf-8
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf


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

    THROTTLING = 1000
    THRESHOLD = 0.999
    STOPPING = 0
    objectives = { "categorical_crossentropy" : "categorical_crossentropy"  }
    OBJECTIVE = objectives[args.loss]

    ########## Load embeddings, dataset and prep input ####################
    #######################################################################
    if EMBEDDING == 'word2vec':
        logging.info("Loading word2vec embeddings")
        word_embeddings = word2vec.load("embeddings/word_embeddings.bin", encoding='ISO-8859-1')
        logging.info("Done loading embeddings!")
        # fit_str = ""
        # for word in word_embeddings.vocab:
            # fit_str += word + " "

    elif EMBEDDING == 'glove':
        word_embeddings = pickle.load(open("embeddings/glove300b.pkl", "rb"))

    elif EMBEDDING == 'rand':
        word_embeddings = {}

    if DATASET == 'semeval':
        from .semeval.prep import Preprocessor, get_dict
        prep = Preprocessor(clipping_value=CLIPPING_VALUE, merge=args.merge)
        output_dict = prep.output_dict
        reverse_dict = prep.reverse_dict 
        dataset_full, labels_full = prep.read_dataset(TRAIN_FILE, debug=DEBUG,
                                                      EXCLUDE_OTHER=EXCLUDE_OTHER)
        #dataset_aug, Y_aug = prep.read_dataset("examples_wiki_with_other", debug=DEBUG)
        #EXCLUDE_OTHER=EXCLUDE_OTHER)

        logging.info("using dataset %s", args.aug_file)
        dataset_aug = prep.read_augset(args.aug_file, debug=DEBUG)
        #dataset_aug = dataset_full[5000:]
        #dataset_full = dataset_full[:5000]
        #labels_full = labels_full[:5000]


    #### Compute number of output classes #################################
    no_of_clz = len(reverse_dict)

    NO_OF_CLASSES = no_of_clz - 1 if args.exclude_other else no_of_clz




    logging.info("#" * 30)
    logging.info("DATA SPLIT AND LOADED")
    logging.info("#" * 30)

    ########

    (word_input_train, nom1_train, nom2_train, 
        e_pairs_train, _, _,
        Y_train) = prep.fit_transform(dataset_full, labels_full)

    Y_train = build_label_representation(Y_train, OBJECTIVE=OBJECTIVE,
                                NO_OF_CLASSES=NO_OF_CLASSES)

    logging.info("Prepping unlableled data...")
    
    ids = [i for i in range(0, len(dataset_aug))]
    
    (word_input_aug, nom_pos_aug_1, nom_pos_aug_2, 
    aug_e_pairs, aug_texts) = prep.fit_transform_aug(dataset_aug)

    X_aug = [word_input_aug, nom_pos_aug_1, nom_pos_aug_2]

    assert len(aug_texts) == len(X_aug[0])
    logging.info("Done prepping unlabeled data..")
    word_index = prep.word_idx()

    logging.info("Ready for Train Val Holdout splits")
    logging.info(str(FOLDS) + " holdout part")

    all_results = []
    
    fold_split = 1.0 / FOLDS
    


    # (word_input_train, 
    #     word_input_test, 
    #     nom1_train, 
    #     nom1_test, 
    #     nom2_train, 
    #     nom2_test,
    #     Y_train,
    #     Y_test,
    #     e_pairs_train,
    #     e_pairs_test ) = train_test_split ( word_input_train, 
    #                                     nom1_train, 
    #                                     nom2_train,
    #                                     Y_train,
    #                                     e_pairs_train,
    #                                     test_size = fold_split )

    (word_input_train,
        word_input_val,
        nom1_train,
        nom1_val,
        nom2_train,
        nom2_val,
        Y_train,
        Y_val,
        e_pairs_train,
        e_pairs_val ) = train_test_split ( word_input_train, 
                                       nom1_train, 
                                       nom2_train, 
                                       Y_train,
                                       e_pairs_train,
                                       test_size = 0.3 )


    X_aug_backup = X_aug

    X_train = [word_input_train, nom1_train, nom2_train]
    X_val = [word_input_val, nom1_val, nom2_val]
    # X_test = [word_input_test, nom1_test, nom2_test]

    # ## 10% val split

    ## KF split returns indices, while testsplit returns values

    result = 0.0
    run_once = False

    # FOr throttling and balancing
    clz_count = defaultdict(int)
    fold_clz_count = defaultdict(int)
    
    for yf in [np.argmax(y_) for y_ in Y_train]:
        fold_clz_count[yf] += 1

    fold_size = Y_train.shape[0]
    distrib_clz = { k : float(v) / float(fold_size)  for k, v in fold_clz_count.items() }

    distrib_throttle = defaultdict(int, { k : math.ceil( v*THROTTLING ) for k,v in distrib_clz.items() })
    prev_loss = 999999999
    while True:              
        model = get_model( 
            word_embeddings=word_embeddings,
            word_index=word_index,
            L2_VALUE=L2_VALUE,
            WINDOW_HEIGHT=WINDOW_HEIGHT,
            DROPOUT_RATE=DROPOUT_RATE,
            WINDOW_SIZE=WINDOW_SIZE,
            loss=OBJECTIVE,
            NO_OF_CLASSES=NO_OF_CLASSES)

        logging.info(model.summary())

        history = train_model(model, X_train, Y_train, EPOCHS, early_stopping=True, val_data=(X_val, Y_val))
        
        X_trainval = [np.append(x, X_val[i], axis=0) for i, x in enumerate(X_train)]
        Y_trainval = np.append(Y_train, Y_val, axis=0)
        


       
        best_epoch = np.argmin(np.asarray(history.history['val_loss'])) + 1

        best_loss = np.asarray(history.history['val_loss']).min()
        logging.info("Training with " + str(best_epoch) + " epochs by early stopping")
        
        K.clear_session()

        model = get_model( 
            word_embeddings=word_embeddings,
            word_index=word_index,
            WINDOW_HEIGHT=WINDOW_HEIGHT,
            DROPOUT_RATE=DROPOUT_RATE,
            WINDOW_SIZE=WINDOW_SIZE,
            loss=OBJECTIVE,
            NO_OF_CLASSES=NO_OF_CLASSES)



        history = train_model(model, X_trainval, Y_trainval, best_epoch, early_stopping=False)


        #logging.info("Evaluating on fold for intermediate result")
        #Y_pred = model.predict(X_test)
        #new_result = f1_score([np.argmax(x) for x in Y_pred],[np.argmax(y) for y in Y_test], average='macro')
        #new_result = model.evaluate(X_test, Y_test)[0]    
        #logging.info("New Loss: %f", new_result)
        #logging.info("#" * 30)
        #if new_result < result:
            #logging.info("No improvement in score, moving to next fold")
            #break

        logging.info("Classifying unlabeled data...")    
        preds = model.predict(X_aug)
        
        logging.info("Selecting samples to include in training set...")

        best_clz = np.asarray([np.argmax(p) for p in preds])
        best_prob = np.asarray([np.max(p) for p in preds])


        # pickle.dump([e_pairs, aug_e_pairs, preds, aug_texts], open("ei.pkl", "wb"))
        # model.save("ei.model")
        correct_preds = 0

        samples_to_add = []
        clz_to_add = []

        ###### Selection of new samples

        # Balanced selection of equal amounts of samples

        for current_clz in range(0, no_of_clz-1, 2):


            clz_threshold = 0.95 if current_clz in [2,6] else THRESHOLD


            brother_clz = current_clz + 1
            # Get all samples of specific class
            clz_preds = np.asarray([idx for idx, clz in enumerate(best_clz) if (clz == current_clz or clz == brother_clz)
                #and aug_e_pairs[idx] in e_pairs_train and best_prob[idx] > clz_threshold])
                and best_prob[idx] > clz_threshold])
            if clz_preds.any():
                no_to_get = THROTTLING
                
                #no_to_get = distrib_throttle[current_clz] 
                if not current_clz == 18:
                      no_to_get += distrib_throttle[brother_clz]

                indices = np.asarray(list(range(0, len(clz_preds))))

                if(len(clz_preds) > THROTTLING):
                   clz_preds = np.random.choice(clz_preds, size=THROTTLING, replace=False)
                
                ## Distribution throttling
                clz_probs = np.asarray(best_prob)[clz_preds]
                
                # best_choices = clz_probs.argsort()[-no_to_get:][::-1]
                #clz_preds = clz_preds[best_choices]
                

                samples_to_add.extend(list(clz_preds))
                clz_to_add.extend(list(best_clz[clz_preds]))
                avg_conf = clz_probs.mean()
                logging.info("Selecting %i samples for training. Best average confidence for class %i is %f", len(clz_preds), current_clz, avg_conf)
            else:
                logging.info("No samples found for class %i", current_clz)
        
        logging.info("Found %i samples to add. ", len(samples_to_add))
        if(len(samples_to_add) <= STOPPING) or prev_loss < best_loss or run_once:
            logging.info("No additional samples, added. Quitting experiment")
            logging.info("Loss %f and previous loss %f", best_loss, prev_loss)
            break

        prev_loss = best_loss
        #true_preds = len([i for i,x in enumerate(clz_to_add) if np.asarray(Y_aug)[samples_to_add][i] == x])
        #logging.info("%i true labels out of %i total selected with accuracy %f", true_preds, len(samples_to_add), true_preds / len(samples_to_add) )
        #result = new_result

        clz_to_add = build_label_representation(np.asarray(clz_to_add), OBJECTIVE=OBJECTIVE,
                            NO_OF_CLASSES=NO_OF_CLASSES,
                            WINDOW_SIZE=WINDOW_SIZE)


        X_train = [np.append(X_train[k], X_aug[k][samples_to_add], axis=0) for k in range(0, len(X_train))]


        ### deleting samples from X_aug
        Y_train = np.append(Y_train, clz_to_add, axis=0)

        import ipdb
        ipdb.sset_trace()
        logging.info("Created new dataset, beginning next iteration")

        X_aug = [np.delete(x, samples_to_add, axis=0) for x in X_aug] 
        
        run_once = True


        
        logging.info("Session cleared")
        

    ########## Writing test ################

    logging.info("Writing test")

    test_path = "data/semeval/testing/" + TEST_FILE + ".txt"
    test_set = open(test_path)
    data, ids = prep.read_testset(test_set, output_dict)

    (X_test, X_test_nom_pos1, X_test_nom_pos2, _, 
        _, _, kept_ids) = prep.fit_transform(data,ids)

    X_test = [X_test, X_test_nom_pos1, X_test_nom_pos2]
    failures = [x for x in ids if x not in kept_ids]

    logging.info("Predicting for X...")    
    preds = model.predict(X_test)

    preds_final = [np.argmax(pred) for pred in preds]

    lookup_labels = [reverse_dict[pred] for pred in preds_final]

    with open("data/semeval/test_pred.txt"  , "w+") as f:
        for idx, i in enumerate(kept_ids):
            f.write(i + "\t" + lookup_labels[idx] + "\n")

        for i in failures:
            f.write(i + "\t" + "Other" + "\n")



    #final_loss = sum([x[0] for x in all_results  ]) / fold 
    # logging.info( "#" * 30)
    # logging.info( "FINAL F1 VALUE: %f", result)
    # logging.info( "#" * 30 )

    logging.info("Experiment done!")

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    main(args)