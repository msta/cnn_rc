
import numpy as np
import tensorflow as tf

from functools import partial

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

from keras.regularizers import l2
from keras.constraints import maxnorm
from keras import backend as K
from functions import fbetascore, margin_loss, MultLayer, accuracy2


def get_model(
    word_embeddings,
    word_index, 
    n, 
    word_entity_dictionary={},
    WORD_EMBEDDING_DIM=300,
    POS_EMBEDDING_DIM=25,
    L2_NORM_MAX=3,
    INCLUDE_POS_EMB=True,
    INCLUDE_ATTENTION=False,
    ACTIVATION_FUNCTION="tanh",
    WINDOW_SIZE=1000,
    optimizer='ada',
    loss=margin_loss,
    DROPOUT_RATE=0.5, 
    NO_OF_CLASSES=19):
    # NOTE TO SELF - Don't let the vector be all-zeroes when the word is not present


    missed_words = 0
    embedding_matrix = np.zeros((len(word_index) + 1, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        try:
            embedding_vector = word_embeddings[word]
        except KeyError:
            missed_words += 1
            #embedding_vector = oov_vector
            embedding_vector = np.random.uniform(-0.25, 0.25, WORD_EMBEDDING_DIM)
        finally:
            embedding_matrix[i] = embedding_vector

    #### attention matrix initialization 
    attention_matrix = np.zeros((len(word_entity_dictionary) + 1, 1))
    for (w,e),idx in word_entity_dictionary.iteritems():
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

    position_embedding = Embedding(2 * n - 1,
                                   POS_EMBEDDING_DIM,
                                   init='zero',
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

    sequence_input = Input(shape=(n,), dtype="int32", name='seq_input')
    position_input_1 = Input(shape=(n,), dtype="int32", name='pos_input1')
    position_input_2 = Input(shape=(n,), dtype="int32", name='pos_input2')


    word_embeddings = embedding_layer(sequence_input)
    position_embeddings_1 = position_embedding(position_input_1)
    position_embeddings_2 = position_embedding(position_input_2)

    
    if INCLUDE_ATTENTION:
        attention_input_1 = Input(shape=(n,), dtype="int32", name='att_input1')
        attention_input_2 = Input(shape=(n,), dtype="int32", name='att_input2')
        attention_score_1 = att_embbeding(attention_input_1)
        attention_score_2 = att_embbeding(attention_input_2)

        attention_score_1 = Activation('softmax', name='att_softmax1')(attention_score_1) 
                                                  
        attention_score_2 = Activation('softmax', name='att_softmax2')(attention_score_2)
                                                  


    if INCLUDE_POS_EMB:
        CIP = WORD_EMBEDDING_DIM + POS_EMBEDDING_DIM * 2
        conv_input = merge([word_embeddings, 
            position_embeddings_1, 
            position_embeddings_2], 
            mode='concat', 
            concat_axis=2,
            name='word+pos_concat')
    else:
        CIP = WORD_EMBEDDING_DIM
        conv_input = word_embeddings


    def att_comp(tensor_list):
        import tensorflow as tf 
        return tf.mul(tensor_list[0],tensor_list[1]) 

    if INCLUDE_ATTENTION:
        ## composition layer
        att_merged = merge([attention_score_1, attention_score_2], 
                            mode="ave", name='attention_mean')
        

        conv_input = merge([att_merged, conv_input], 
                            mode=att_comp,
                            output_shape=(n, CIP),
                            name='att_composition')


    ## activation function according to paper
    g = ACTIVATION_FUNCTION

    #windows = [2,3,4,5]
    windows = [3]

    p_list = []

    for w in windows:
        conv = conv_input   
        #conv = Reshape((1,n,CIP))(conv_input)
        conv = Conv1D(WINDOW_SIZE, w, 
            border_mode='same',
            activation=g,
            W_constraint=maxnorm(L2_NORM_MAX), 
            bias=True,
            init='glorot_normal',
            name='r_convolved')(conv)
        #conv = GlobalMaxPooling2D()(conv)
        p_list.append(conv)

    convolved = p_list[0]
    network = Reshape((WINDOW_SIZE, n), name='r_transposed')(convolved)
    

    # network = merge([convolved, corr_matrix],
    #                     mode=att_comp2,
    #                     output_shape=(n, NO_OF_CLASSES))

    #### Multiply RT with U and the EMBEDDING 

    ##### The learned class embedding ######
    final = Dense(NO_OF_CLASSES, bias=False, name='U')(network)
    class_embedding = Dense(WINDOW_SIZE, bias=False, name='WL')


    final = class_embedding(final)


    G = Activation('softmax')(final)



    #final = MaxPooling1D(pool_length=WINDOW_SIZE/100)(final)

    #final = Flatten()(final)
    # final = Dropout(DROPOUT_RATE)(final)
    #final = Dense(NO_OF_CLASSES, activation='softmax')(final)

    rap = merge([convolved, G], mode='dot', output_shape=(n,WINDOW_SIZE))

    ### This has dimensions window_size
    final = GlobalMaxPooling1D()(rap)

    input_arr = [sequence_input]
    
    if INCLUDE_POS_EMB:
        input_arr.append(position_input_1)
        input_arr.append(position_input_2)

    if INCLUDE_ATTENTION:
        input_arr.append(attention_input_1)
        input_arr.append(attention_input_2)

    #### Scoring crap, DOESNT WORK
    # final_t = Reshape((1, WINDOW_SIZE))(final)
    # weights_as_batch = K.cast(class_embedding.weights[0], K.floatx())
    # weights_as_batch = tf.reshape(class_embedding.weights[0], [-1, NO_OF_CLASSES, WINDOW_SIZE])
    # final = merge([final_t, weights_as_batch], mode='dot', dot_axes=(2,2))
    # final = Flatten()(final)
    ####


    ###### We do a partial trick to include the class embeddings and calculate the accuracy

    acc_partial = partial(accuracy2, class_embedding.weights[0])
    acc_partial.__name__='accuracy'

    loss = partial(margin_loss, class_embedding.weights[0]) if loss == margin_loss else loss
    model = Model(input=input_arr, output=[final])
    
    if optimizer == 'ada':
        opt = Adadelta(epsilon=1e-06)
    ### from the att matrix paper
    elif optimizer == 'sgd':
        opt = SGD(lr=0.03)

    model.compile(optimizer=opt, loss=loss, metrics=[acc_partial])
    return model
