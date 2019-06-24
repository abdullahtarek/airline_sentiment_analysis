import tensorflow as tf

def make_model(maxSequence ,wordembedding_len,numClasses,input_data,keep_prob, keep_prob2 ):

    lstm_cell = tf.nn.rnn_cell.GRUCell(128 , activation="relu")
    lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,state_keep_prob=keep_prob2, output_keep_prob=keep_prob)


    lstm_cell2 = tf.nn.rnn_cell.GRUCell(64 , activation="relu")
    lstm_dropout2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, state_keep_prob=keep_prob2, output_keep_prob=keep_prob)


    #lstm_cell3 = tf.nn.rnn_cell.BasicRNNCell(32)
    #lstm_dropout3 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell3,state_keep_prob=keep_prob2, output_keep_prob=keep_prob)


    lstm_layers =  tf.contrib.rnn.MultiRNNCell([ lstm_cell2])

    value, states = tf.nn.dynamic_rnn(lstm_layers, input_data, dtype=tf.float32)
    
    weight = tf.Variable(tf.truncated_normal([64, numClasses]) , name="w0")
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]),name="b0")
    value = tf.transpose(value, [1, 0, 2],name="t0")
    last = tf.gather(value, int(value.get_shape()[0]) - 1 ,name="g0")
    prediction = tf.add(tf.matmul(last, weight), bias ,name="output" )
    
    return prediction