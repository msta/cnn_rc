import keras.backend as K

def f1_macro(y_true, y_pred):
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0


    beta = 1
    p = precision_macro(y_true, y_pred)
    r = recall_macro(y_true, y_pred)

    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return K.mean(fbeta_score)



def precision_macro(y_true, y_pred):
    true_positives = K.round(K.clip(y_true * y_pred, 0, 1))
    predicted_positives = K.round(K.clip(y_pred, 0, 1))
    precision_macro = (K.epsilon() + true_positives) / (predicted_positives + K.epsilon())
    return precision_macro


def recall_macro(y_true, y_pred):
    true_positives = K.round(K.clip(y_true * y_pred, 0, 1))
    possible_positives = K.round(K.clip(y_true, 0, 1))
    recall_macro = (K.epsilon() + true_positives) / (possible_positives + K.epsilon())
    return recall_macro