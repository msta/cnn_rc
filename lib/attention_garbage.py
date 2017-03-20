
''' NOVEL DISTANCE FUNCTION EH? '''
def new_dist(actual, embedding):

    import ipdb
    ipdb.sset_trace()

    actual_sum = tf.reduce_sum(actual, keep_dims=True)
    actual_unit = actual / actual_sum

    embedding_sum = tf.reduce_sum(embedding, keep_dims=True)
    embedding_unit = embedding / embedding_sum


    result = tf.sqrt(tf.reduce_sum(K.square(actual_unit - embedding_unit), 2, keep_dims=True))
    return result


def new_dist3(pred, embedding):
    import ipdb
    ipdb.sset_trace()

    pred_sum = tf.reduce_sum(pred, 1, keep_dims=True)
    pred_unit = pred / pred_sum

    embedding_sum = tf.reduce_sum(embedding)
    embedding_unit = embedding / embedding_sum

    internal_dist = pred_unit - embedding_unit  

    squared = K.square(internal_dist)    

    sum_for_dist = tf.reduce_sum(squared, 1, keep_dims=True)

    result = tf.sqrt(sum_for_dist)
    return result


### TODO test this shit!!!! :D ###
def margin_loss(weights, y_true, y_pred):


    nb_clz = weights.get_shape()[0]

    import ipdb
    ipdb.sset_trace()

    distances = new_dist3(y_pred, weights)

    # index_mask = tf.reshape(tf.one_hot(y_true, nb_clz), [-1,nb_clz])

    true_pred = tf.reduce_sum(distances * index_mask,1)

#    partial_dist = partial(new_dist, embedding=weights)
 #   distances = tf.map_fn(partial_dist, y_pred, dtype=tf.float32)


    #correct = tf.gather(w, y_true)
    # tf.cast(y_true, dtype=tf.int32)
    # correct_embeddings = tf.gather(weights, tf.cast(    y_true, dtype=tf.int32))

    best_incorrect_dist_idx = tf.argmax(distances,1)
    incorrect_dist = distances[1] 

    true_dist = new_dist(correct,y_pred)


    return 1.0 + true_dist - incorrect_dist



def new_dist2(weights):
    
    actual_sum = tf.reduce_sum(actual, 1, keep_dims=True)
    unit_actual = actual / actual_sum
    return tf.sqrt(tf.reduce_sum(K.square(unit_actual - weights), 1, keep_dims=True))



def ranking_loss(y_true, y_pred):

    y = 2   
    m_plus = 2.5
    m_minus = 0.5

    correct_matrix = (y_true * y_pred)

    correct_score = tf.reduce_max(correct_matrix,1)
    incorrect_score = tf.reduce_max(y_pred - correct_matrix, 1)

    return tf.log(1 + tf.exp(y*(m_plus - correct_score))) +  tf.log(1 + tf.exp(y*(m_minus + incorrect_score)))



''' accuracy that chooses a class from the class embedding 
and compares with the categorical cross-entropy '''
def accuracy2(class_emb, y_true, y_pred):
    
    # y_pred_max = K.argmax(K.dot(y_pred, K.transpose(class_emb)), axis=-1)
    y_pred_max = K.argmax(K.dot(K.transpose(y_pred), class_emb), axis=-1)


    y_true_max = K.argmax(y_true, axis=-1)

    acc = K.mean(K.equal(y_true_max, y_pred_max))

    return acc
