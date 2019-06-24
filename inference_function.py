import cv2
import numpy as np
import tensorflow as tf
from word2vec import convert_to_vector , read_word2vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def inference(text_input,word_embedding_path="glove.6B.300d.txt", maxSequence =31):
    word_embedding = read_word2vec(word_embedding_path)  
    text = convert_to_vector(text_input, word_embedding,maxSequence )
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('Best_model/checkpoint_5.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('Best_model/'))
    
    with tf.Session() as sess:    
        saver = tf.train.import_meta_graph('ckpt/checkpoint_5.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('ckpt/'))

        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_data:0")
        keep_prob = graph.get_tensor_by_name("keep_prob1:0")
        keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")

        op_to_restore = graph.get_tensor_by_name("output:0")


        predict =sess.run(op_to_restore,feed_dict={input_x: np.expand_dims(text,axis=0 ),
                                    keep_prob:1 ,
                                    keep_prob2:1 })
        sentiments = ["negative", "nuetral","positive"]
        
        print("The predicted class is : {}".format(sentiments[softmax(predict).argmax()]))