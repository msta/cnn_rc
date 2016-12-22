import re
import logging
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations
from functools import partial
import numpy as np


from keras import backend as K


''' courtesy of keras.io '''
def fbetascore(y_true, y_pred, beta=1):
    import ipdb
    ipdb.sset_trace()
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
def new_dist(actual, weights):

    actual_sum = tf.reduce_sum(actual, keep_dims=True)
    unit_actual = actual / actual_sum
    result = tf.sqrt(tf.reduce_sum(K.square(unit_actual - weights), 1, keep_dims=True))
    return result


def att_comp2(tensor_list):
    from keras import backend as K
    import tensorflow as tf
    return K.batch_dot(tensor_list[0],tensor_list[1]) 




### TODO test this shit!!!! :D ###
def margin_loss(weights, label, actual):
    weights = weights[0]
    partial_dist = partial(new_dist, weights=weights)


    distances = tf.map_fn(partial_dist, actual, dtype=tf.float32)

    
    #correct = tf.gather(w, label)
    # tf.cast(label, dtype=tf.int32)
    correct_embeddings = tf.gather(weights, tf.cast(label, dtype=tf.int32))


    best_incorrect_dist_idx = tf.argmax(distances,1)
    incorrect_dist = distances[1] 

    true_dist = new_dist(correct,actual)


    return 1.0 + true_dist - incorrect_dist



def new_dist2(weights):
    
    actual_sum = tf.reduce_sum(actual, 1, keep_dims=True)
    unit_actual = actual / actual_sum
    return tf.sqrt(tf.reduce_sum(K.square(unit_actual - weights), 1, keep_dims=True))


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
        import ipdb
        ipdb.sset_trace()
        return (input_shape[0], self.output_dim)
