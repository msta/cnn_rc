
import numpy as np

from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Input, merge, Flatten, Reshape
from keras.layers import Convolution2D as Conv2D
from keras.layers.core import Dropout
from keras.layers.pooling import GlobalMaxPooling2D
from keras.regularizers import l2

L2_RATE = 3


def get_model(word_embeddings, 
    word_index, 
    n, 
    WORD_EMBEDDING_DIM,
    INCLUDE_POS_EMB,
    DROPOUT_RATE, 
    NO_OF_CLASSES, 
    oov_vector):
    # NOTE TO SELF - Don't let the vector be all-zeroes when the word is not present

    missed_words = 0
    embedding_matrix = np.zeros((len(word_index) + 1, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        try:
            embedding_vector = word_embeddings[word]
        except KeyError:
            missed_words += 1
            embedding_vector = oov_vector
        finally:
            embedding_matrix[i] = embedding_vector

    ### trainable according to paper
    embedding_layer = Embedding(len(word_index) + 1,
                                WORD_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=n,
                                trainable=False)

    ## check that what happens with non matches in the paper and amount
    ## try dropout word embeddings

    print "#" * 30
    print "WORD EMBEDDINGS LOADED"
    print "DIMENSION : ", len(word_index) + 1, " X ", WORD_EMBEDDING_DIM
    print "WORDS MISSED IN EMBEDDINGS : ", missed_words
    print "#" * 30

    # In[302]:

    POS_EMBEDDING_DIM = 50

    ## Prepare word position embedding
    position_embedding = Embedding(2 * n - 1,
                                   POS_EMBEDDING_DIM,
                                   init='glorot_normal',
                                   input_length=n,
                                   trainable=True)

    print "#" * 30
    print "POSITIONAL EMBEDDINGS LOADED"
    print "#" * 30
    sequence_input = Input(shape=(n,), dtype="int32")
    position_input_1 = Input(shape=(n,), dtype="int32")
    position_input_2 = Input(shape=(n,), dtype="int32")
    word_embeddings = embedding_layer(sequence_input)
    position_embeddings_1 = position_embedding(position_input_1)
    position_embeddings_2 = position_embedding(position_input_2)

    

    

    if INCLUDE_POS_EMB:
        CIP = WORD_EMBEDDING_DIM + POS_EMBEDDING_DIM * 2
        conv_input = merge([word_embeddings, position_embeddings_1, position_embeddings_2 ], mode='concat', concat_axis=2)
    else:
        CIP = WORD_EMBEDDING_DIM
        conv_input = word_embeddings
    ## activation function according to paper
    g = "tanh"

    windows = [2,3,4,5]
    #windows = [2]

    p_list = []

    for w in windows:
        reshaped = Reshape((1,n,CIP))(conv_input)
        window = Conv2D(150,1, w, border_mode='valid',activation=g, bias=True)(reshaped)
        pool = GlobalMaxPooling2D()(window)
        p_list.append(pool)

    #pooling_concat = p_list[0]
    #pooling_concat = conv_input
    pooling_concat = merge(p_list, mode="concat", concat_axis=1)

    # print pooling_concat
    #pooling_concat = Flatten()(pooling_concat)
    pooling_concat = Dropout(DROPOUT_RATE)(pooling_concat)


#    final_layer = Dense(NO_OF_CLASSES, W_regularizer=l2(L2_RATE), activation='softmax')(pooling_concat)
    final_layer = Dense(NO_OF_CLASSES, activation='softmax')(pooling_concat)

    print "#" * 30
    print "NETWORK INITIALIZED"
    print "#" * 30
    #model = Model(input=[sequence_input, position_input], output=[final_layer])
    input_arr = [sequence_input]
    if INCLUDE_POS_EMB:
        input_arr.append(position_input_1, position_input_2) 
    model = Model(input=input_arr, output=[final_layer])
    
    opt_ada = Adadelta()

    model.compile(optimizer=opt_ada, loss='categorical_crossentropy', metrics=["acc"])
    return model
