
import numpy as np
import tensorflow as tf
import logging

from functools import partial

from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, SGD
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Input, merge, Flatten, Reshape
from keras.layers import Merge, Lambda
from keras.layers import Convolution2D as Conv2D
from keras.layers import Convolution1D as Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dropout
from keras.layers.pooling import GlobalMaxPooling2D, GlobalMaxPooling1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.callbacks import EarlyStopping


from keras.regularizers import l2
from keras.constraints import maxnorm
from keras import backend as K
from .metrics import f1_macro

def get_pretrained_model(
    word_embeddings,
    n, 
    WORD_EMBEDDING_DIM=300,
    POS_EMBEDDING_DIM=50,
    L2_NORM_MAX=3,
    L2_VALUE=0.0,
    WINDOW_HEIGHT=[3],
    ACTIVATION_FUNCTION="tanh",
    WINDOW_SIZE=1000,
    optimizer='ada',
    loss='categorical_crossentropy',
    DROPOUT_RATE=0.5, 
    NO_OF_CLASSES=19,
    all_weights=[]):
    # NOTE TO SELF - Don't let the vector be all-zeroes when the word is not present

    INCLUDE_POS_EMB = True if POS_EMBEDDING_DIM > 0 else False

    markup_vector = np.random.uniform(-0.25, 0.25, WORD_EMBEDDING_DIM)

    weight_count = 0


    import ipdb
    ipdb.sset_trace()

    embedding_layer = Embedding(all_weights[weight_count].shape[0],
                                WORD_EMBEDDING_DIM,
                                weights=[all_weights[weight_count]],
                                input_length=n,
                                name="word_embedding",
                                trainable=False)
    weight_count += 1

    ## Removed -1 to keep a special token, 0, for padding.
    position_embedding = Embedding(2 * n,
                                   POS_EMBEDDING_DIM,
                                   W_constraint=maxnorm(L2_NORM_MAX),
                                   input_length=n,
                                   weights=[all_weights[weight_count]],
                                   name='pos_embedding',
                                   trainable=True)

    weight_count += 1


    sequence_input = Input(shape=(n,), dtype="int32", name='seq_input')
    position_input_1 = Input(shape=(n,), dtype="int32", name='pos_input1')
    position_input_2 = Input(shape=(n,), dtype="int32", name='pos_input2')

    # sequence_input_paths = Input(shape=(n,), dtype="int32", name='seq_input1')
    # position_input_paths_1 = Input(shape=(n,), dtype="int32", name='pos_input11')
    # position_input_paths_2 = Input(shape=(n,), dtype="int32", name='pos_input22')

    word_embeddings = embedding_layer(sequence_input)
    position_embeddings_1 = position_embedding(position_input_1)
    position_embeddings_2 = position_embedding(position_input_2)

                                                  
    CIP = WORD_EMBEDDING_DIM
    embedding_list = [word_embeddings]

    if INCLUDE_POS_EMB:
        CIP += (POS_EMBEDDING_DIM * 2)
        embedding_list.append(position_embeddings_1)
        embedding_list.append(position_embeddings_2)
        
    if len(embedding_list) > 1:
        conv_input = merge(embedding_list, 
                mode='concat', 
                concat_axis=2,
                name='embedding_merge_1')
    else:
        conv_input = embedding_list[0]


    ## activation function according to paper
    g = ACTIVATION_FUNCTION

    #windows = [2,3,4,5]
    windows = WINDOW_HEIGHT

    p_list = []

    for w in windows:
        conv = conv_input   
        #conv = Reshape((1,n,CIP))(conv_input)
        import ipdb
        ipdb.sset_trace()
        conv_weights = [all_weights[weight_count], all_weights[weight_count + 1]]
        weight_count += 2         

        conv = Conv1D(WINDOW_SIZE, w, 
            border_mode='valid',
            activation=g,
            bias=True,
            init='glorot_normal',
            weights=conv_weights,
            W_constraint=maxnorm(L2_NORM_MAX),
            name='r_convolved' + str(w))(conv)


        conv = GlobalMaxPooling1D()(conv)
        p_list.append(conv)

    if len(windows) > 1:
        convolved = merge(p_list, mode='concat', concat_axis=1)
    else:
        convolved = p_list[0]


    final = build_nguyen_cnn(convolved,
                             DROPOUT_RATE,
                             L2_NORM_MAX,
                             L2_VALUE,
                             NO_OF_CLASSES,
                             weights=[all_weights[weight_count], all_weights[weight_count + 1]])


    input_arr = [sequence_input]
    
    if INCLUDE_POS_EMB:
        input_arr.append(position_input_1)
        input_arr.append(position_input_2)


    loss = build_loss(loss)
    optimizer = build_optimizer(optimizer)
    metrics = build_metrics()


    model = Model(input=input_arr, output=[final])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def build_nguyen_cnn(convolved, DROPOUT_RATE, L2_NORM_MAX, 
                     L2_VALUE, NO_OF_CLASSES, weights=[]):
    
    #convolved = GlobalMaxPooling1D()(convolved)
    #dropout = pooled
    convolved = Dropout(DROPOUT_RATE)(convolved)
    output = Dense(NO_OF_CLASSES, 
                    init='glorot_uniform',
                    W_regularizer=l2(L2_VALUE),
                    weights=weights,
                    W_constraint=maxnorm(L2_NORM_MAX),
                    activation='softmax')(convolved)
    return output


def build_loss(loss):
    return loss

def build_metrics(with_attention_two=False):
    metrics = []
    if with_attention_two:
        acc_partial = partial(accuracy2, class_embedding.weights[0])
        acc_partial.__name__='accuracy'
        metrics.append(acc_partial)
    else:
        metrics.extend(['fmeasure', 'accuracy'])
    return metrics


def build_optimizer(optimizer):
    if optimizer == 'ada':
        opt = Adadelta(epsilon=1e-06)
    elif optimizer == 'sgd':
        opt = SGD(lr=0.05, decay=0.00)
    return opt


def train_model(model, X_train, Y_train, EPOCHS, early_stopping=False):
                #logging.debug(model.summary())
                if early_stopping:
                    history = model.fit(X_train,
                        Y_train, 
                        validation_split=0.1,
                        callbacks=[EarlyStopping(patience=1)],
                        nb_epoch=EPOCHS, 
                        batch_size=50, 
                        shuffle=True)
                else:
                    history = model.fit(X_train,
                    Y_train, 
                    nb_epoch=EPOCHS, 
                    batch_size=50, 
                    shuffle=True)
                return history



#### Scoring crap, DOESNT WORK
# final_t = Reshape((1, WINDOW_SIZE))(final)
# weights_as_batch = K.cast(class_embedding.weights[0], K.floatx())
# weights_as_batch = tf.reshape(class_embedding.weights[0], [-1, NO_OF_CLASSES, WINDOW_SIZE])
# final = merge([final_t, weights_as_batch], mode='dot', dot_axes=(2,2))
# final = Flatten()(final)
####
