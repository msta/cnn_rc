
import numpy as np
import tensorflow as tf
import logging

from functools import partial

from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, SGD
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Input, Flatten, Reshape
from keras.layers import Merge, Lambda
from keras.layers import Convolution2D as Conv2D
from keras.layers import Convolution1D as Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dropout
from keras.layers.pooling import GlobalMaxPooling2D, GlobalMaxPooling1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import Dot, Concatenate, Average
from keras.callbacks import EarlyStopping


from keras.regularizers import l2
from keras.constraints import maxnorm
from keras import backend as K
from .metrics import f1_macro

def get_model(
    word_embeddings,
    word_index, 
    n, 
    word_entity_dictionary={},
    WORD_EMBEDDING_DIM=300,
    POS_EMBEDDING_DIM=50,
    L2_NORM_MAX=3,
    L2_VALUE=0.0,
    WORDNET=False,
    WORDNET_LEN=0,
    WINDOW_HEIGHT=[3],
    INCLUDE_ATTENTION_ONE=False,
    INCLUDE_ATTENTION_TWO=False,
    ACTIVATION_FUNCTION="relu",
    WINDOW_SIZE=150,
    optimizer='ada',
    loss='categorical_crossentropy',
    DROPOUT_RATE=0.5, 
    NO_OF_CLASSES=19):
    # NOTE TO SELF - Don't let the vector be all-zeroes when the word is not present

    INCLUDE_POS_EMB = True if POS_EMBEDDING_DIM > 0 else False

    missed_words = []
    embedding_matrix = np.zeros((len(word_index) + 1, WORD_EMBEDDING_DIM))

    markup_vector = np.random.uniform(-0.25, 0.25, WORD_EMBEDDING_DIM)

    word_index_list = list(word_index.items())

    for word, i in word_index_list:
        try:
            #if word in ['e1', 'e2']:
            #    embedding_vector = markup_vector
            #else:
            embedding_vector = word_embeddings[word]
        except KeyError:
            try:
                embedding_vector = word_embeddings[word.lower()]
            except KeyError:
                missed_words.append(word)
                #embedding_vector = oov_vector
                embedding_vector = np.random.uniform(-0.25, 0.25, WORD_EMBEDDING_DIM)
        finally:
            embedding_matrix[i] = embedding_vector
    logging.info("Missed words" + str(len(missed_words)))

    #### attention matrix initialization 

    attention_matrix = np.zeros((len(word_entity_dictionary) + 1, 1))
    for (w,e),idx in word_entity_dictionary.items():
        emb1 = embedding_matrix[w]
        emb2 = embedding_matrix[e]
        a_val = np.inner(emb1, emb2)
        attention_matrix[idx] = a_val


    embedding_layer = Embedding(len(word_index) + 1,
                                WORD_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=n,
                                name="word_embedding",
                                trainable=True)

    ## Removed -1 to keep a special token, 0, for padding.
    position_embedding = Embedding(2 * n,
                                   POS_EMBEDDING_DIM,
                                   embeddings_constraint=maxnorm(L2_NORM_MAX),
                                   embeddings_initializer='glorot_uniform',
                                   input_length=n,
                                   name='pos_embedding',
                                   trainable=True)

    ### Attention matrice
    att_embbeding = Embedding(len(word_entity_dictionary)+1,
                                1,
                                weights=[attention_matrix],
                                input_length=n,
                                name='A_word_pairs',
                                trainable=True)

    wordnet_emb = Embedding(WORDNET_LEN,
                            POS_EMBEDDING_DIM,
                            input_length=n,
                            embeddings_initializer='glorot_uniform',
                            name='wordnet_emb',
                            trainable=True)

    sequence_input = Input(shape=(n,), dtype="int32", name='seq_input')
    position_input_1 = Input(shape=(n,), dtype="int32", name='pos_input1')
    position_input_2 = Input(shape=(n,), dtype="int32", name='pos_input2')
    wordnet_input = Input(shape=(n,), dtype="int32", name='wordnet_input')

    # sequence_input_paths = Input(shape=(n,), dtype="int32", name='seq_input1')
    # position_input_paths_1 = Input(shape=(n,), dtype="int32", name='pos_input11')
    # position_input_paths_2 = Input(shape=(n,), dtype="int32", name='pos_input22')

    word_embeddings = embedding_layer(sequence_input)
    position_embeddings_1 = position_embedding(position_input_1)
    position_embeddings_2 = position_embedding(position_input_2)
    wordnet_embedding = wordnet_emb(wordnet_input)

    if INCLUDE_ATTENTION_ONE:
        attention_input_1 = Input(shape=(n,), dtype="int32", name='att_input1')
        attention_input_2 = Input(shape=(n,), dtype="int32", name='att_input2')
        attention_score_1 = att_embbeding(attention_input_1)
        attention_score_2 = att_embbeding(attention_input_2)

        attention_score_1 = Activation('softmax', name='att_softmax1')(attention_score_1) 
                                                  
        attention_score_2 = Activation('softmax', name='att_softmax2')(attention_score_2)
                                                  
    CIP = WORD_EMBEDDING_DIM
    embedding_list = [word_embeddings]

    if INCLUDE_POS_EMB:
        CIP += (POS_EMBEDDING_DIM * 2)
        embedding_list.append(position_embeddings_1)
        embedding_list.append(position_embeddings_2)
        
    if WORDNET:
        CIP += POS_EMBEDDING_DIM
        embedding_list.append(wordnet_embedding)
        

    if len(embedding_list) > 1:
        conv_input = Concatenate(name='embedding_merge_1', axis=2)(embedding_list)
    else:
        conv_input = embedding_list[0]


    if INCLUDE_ATTENTION_ONE:
        ## composition layer
        pass
        # att_merged = merge([attention_score_1, attention_score_2], 
        #                     mode="ave", name='attention_mean')
        

        # conv_input = merge([att_merged, conv_input], 
        #                     mode=att_comp,
        #                     output_shape=(n, CIP),
        #                     name='att_composition')


    ## activation function according to paper
    g = ACTIVATION_FUNCTION

    #windows = [2,3,4,5]
    windows = WINDOW_HEIGHT

    p_list = []

    for w in windows:
        conv = conv_input   
        #conv = Reshape((1,n,CIP))(conv_input)
        conv = Conv1D(WINDOW_SIZE, w, 
            padding='valid',
            activation=g,
            use_bias=True,
            kernel_constraint=maxnorm(L2_NORM_MAX),
            name='r_convolved' + str(w))(conv)
        conv = GlobalMaxPooling1D()(conv)
        p_list.append(conv)

    if len(windows) > 1:
        convolved = Concatenate(axis=1)(p_list)
    else:
        convolved = p_list[0]

    
    if INCLUDE_ATTENTION_TWO:
        final = build_attention_two(convolved,
                                    NO_OF_CLASSES,
                                    WINDOW_SIZE)
        assert INCLUDE_ATTENTION_ONE

    else:
        final = build_nguyen_cnn(convolved,
                                 DROPOUT_RATE,
                                 L2_NORM_MAX,
                                 L2_VALUE,
                                 NO_OF_CLASSES)


    input_arr = [sequence_input]
    
    if INCLUDE_POS_EMB:
        input_arr.append(position_input_1)
        input_arr.append(position_input_2)

    if WORDNET:
        input_arr.append(wordnet_input)

    if INCLUDE_ATTENTION_ONE:
        input_arr.append(attention_input_1)
        input_arr.append(attention_input_2)



    loss = build_loss(loss, INCLUDE_ATTENTION_TWO)
    optimizer = build_optimizer(optimizer)
    metrics = build_metrics(INCLUDE_ATTENTION_TWO)


    model = Model(inputs=input_arr, outputs=[final])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def build_nguyen_cnn(convolved, DROPOUT_RATE, L2_NORM_MAX, 
                     L2_VALUE, NO_OF_CLASSES):
    
    #convolved = GlobalMaxPooling1D()(convolved)
    #dropout = pooled
    convolved = Dropout(DROPOUT_RATE)(convolved)
    output = Dense(NO_OF_CLASSES, 
                    kernel_constraint=maxnorm(L2_NORM_MAX),
                    activation='softmax')(convolved)
    return output


def build_loss(loss, INCLUDE_ATTENTION_TWO):
    if loss == 'margin_loss':
        assert INCLUDE_ATTENTION_TWO
        loss = partial(margin_loss, class_embedding.weights[0])
    return loss

def build_metrics(with_attention_two=False):
    metrics = []
    if with_attention_two:
        acc_partial = partial(accuracy2, class_embedding.weights[0])
        acc_partial.__name__='accuracy'
        metrics.append(acc_partial)
    else:
        metrics.extend(['accuracy'])
    return metrics


def build_optimizer(optimizer):
    if optimizer == 'ada':
        opt = Adadelta(epsilon=1e-06)
    elif optimizer == 'sgd':
        opt = SGD(lr=0.05, decay=0.00)
    return opt

def build_attention_two(convolved,
                        NO_OF_CLASSES,
                        WINDOW_SIZE):
    # Transpose R
    network = Reshape((WINDOW_SIZE, n), name='r_transposed')(convolved)
    # U matrix before Weight Embedding
    final = Dense(NO_OF_CLASSES, bias=False, name='U')(network)
    # WL aka Weight Embedding
    class_embedding = Dense(WINDOW_SIZE, bias=False, name='WL')
    final = class_embedding(final)

    # Apply softmax to get AP 
    AP = Activation('softmax')(final)

    # Multiply RT with AP to get highlight phrase-level components 
    #rap = merge([convolved, AP], mode='dot', output_shape=(n,WINDOW_SIZE))
    rap = None
    ### Obtain wO which approximates a column in WL
    final = GlobalMaxPooling1D()(rap)
    return final

def att_comp(tensor_list):
    return tf.mul(tensor_list[0],tensor_list[1]) 


def train_model(model, X_train, Y_train, EPOCHS, early_stopping=False, val_data=None):
                #logging.debug(model.summary())
                if early_stopping:
                    assert val_data
                    history = model.fit(X_train,
                        Y_train, 
                        validation_data=val_data,
                        callbacks=[EarlyStopping(patience=1)],
                        epochs=EPOCHS, 
                        batch_size=50, 
                        shuffle=True)
                else:
                    history = model.fit(X_train,
                    Y_train, 
                    epochs=EPOCHS, 
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
