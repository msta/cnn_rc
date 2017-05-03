import re
import logging
import numpy as np

from functools import partial


from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer

    

def debug_print_dict(input, msg):
    logging.debug("#" * 30)
    logging.debug(msg)
    for k, v in input.items():
        logging.debug(str(k) + " : " + str(v))
    logging.debug(msg + " length: ")
    logging.debug(len(input))
    logging.debug("#" * 30)

def debug_print(input, msg):
    logging.debug("#" * 30)
    logging.debug(msg)
    for line in input:
        logging.debug(line)
    logging.debug("#" * 30)

def clean_classes(dict):
    clean = set([x.split("(")[0] for x in dict])
    count = 0
    o = {}
    for c in clean:
        o[c] = count
        count += 1
    return o

def transform_to_embedding(label, nb_classes, dimensions):
    m = np.zeros((nb_classes, dimensions))
    m[label,:] = 1
    return m


def att_comp2(tensor_list):
    return K.batch_dot(tensor_list[0],tensor_list[1]) 


def build_input_arrays_test(X, 
                            X_nom_pos1, X_nom_pos2):
    if X_nom_pos1.any():
        X.append(X_nom_pos1)
        X.append(X_nom_pos2)



def build_input_arrays_test(X,
                            att_list_1, att_list_2, 
                            X_nom_pos1, X_nom_pos2,
                            wordnet_sequences):
    if X_nom_pos1.any():
        X.append(X_nom_pos1)
        X.append(X_nom_pos2)
    if np.asarray(wordnet_sequences).any():
        X.append(wordnet_sequences)

    if np.asarray(att_list_1).any():
        X.append(att_list_1)
        X.append(att_list_2)



def build_label_representation(Y_train, OBJECTIVE=None, NO_OF_CLASSES=19, WINDOW_SIZE=150):

    if OBJECTIVE in ['categorical_crossentropy', 'margin_loss']:
        Y_train = to_categorical(Y_train, num_classes=NO_OF_CLASSES)
        # elif loss == margin_loss:
        #     Y_train = [transform_to_embedding(label, NO_OF_CLASSES, WINDOW_SIZE) for label in Y_train]
        #     Y_test = [transform_to_embedding(label, NO_OF_CLASSES, WINDOW_SIZE) for label in Y_test]
    else:
        raise ValueError
    return Y_train
