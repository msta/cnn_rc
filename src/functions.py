import re
import logging
import numpy as np
import tensorflow as tf

from keras import backend as K


''' courtesy of keras.io '''
def fbetascore(y_true, y_pred, beta=1):
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

def debug_print_dict(input, msg):
    logging.debug("#" * 30)
    logging.debug(msg)
    for k, v in input.iteritems():
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

def process_train(line):
    # split on quotes and get the raw text
    stripped = re.sub("\d{0,4}", "", line, count=1).strip()
    return stripped[1:len(stripped)-2]

''' NOVEL DISTANCE FUNCTION EH? '''
def new_dist(relation, actual):
    actual_sum = tf.reduce_sum(actual, 1, keep_dims=True)
    unit_actual = actual / actual_sum
    return tf.sqrt(tf.reduce_sum(K.square(unit_actual - relation), 1, keep_dims=True))


### TODO test this shit!!!! :D ###
def margin_loss(true, actual):
    ## here the relation is the ground truth label
    true_dist = new_dist(true,actual)
    # hack to find the arg max!
    incorrect = tf.argmax((actual-true-true), 1)
    inc_one_hots = tf.one_hot(incorrect, actual.get_shape()[1])    
    incorrect_dist = new_dist(inc_one_hots, actual)
    return 1.0 + true_dist - incorrect_dist


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations
import tensorflow as tf
import numpy as np

class MultLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.init = initializations.get('glorot_uniform')
        self.output_dim = output_dim[1]
        super(MultLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        initial_weight_value = np.random.random((input_dim, self.output_dim))
        
        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))

        self.trainable_weights = [self.W]
        super(MultLayer, self).build(input_shape)  # be sure you call this somewhere

    def call(self, x, mask=None):
        print "hej"
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):

        return (input_shape[0], self.output_dim)
