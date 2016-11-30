import re

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




def debug_print(input, msg):
    print "#" * 30
    print msg
    for line in input:
        print line
    print "#" * 30

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


