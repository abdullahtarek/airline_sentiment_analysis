import tensorflow as tf
from model import make_model
import numpy as np
import random
import os

#stratified Btaches
def get_next_batch(batch_size, input_data,labels):
    counter=0
    classes = [[1,0,0],[0,1,0],[0,0,1]]
    batch_data = []
    batch_labels=[]
    for i in range(batch_size):
        indecies = np.where(np.all(labels ==classes[counter], axis=1))
        index = random.randint(0,indecies[0].shape[0]-1)
        batch_data.append(input_data[indecies[0][index]])
        batch_labels.append(labels[indecies[0][index]])
        
        counter+=1
        if counter >2:
            counter=0
    return  np.array(batch_data), np.array(batch_labels)


def train(maxSequence ,wordembedding_len,numClasses,num_of_epochs , batch_size , train_data,train_labels,val_data, val_labels ):
    tf.reset_default_graph()
    keep_prob =  tf.placeholder(tf.float32,name="keep_prob1")
    keep_prob2 =  tf.placeholder(tf.float32,name="keep_prob2")
    labels = tf.placeholder(tf.float32, [None, numClasses],name="labels")
    input_data = tf.placeholder(tf.float32, [None, maxSequence,wordembedding_len], name="input_data")

    prediction = make_model(maxSequence ,wordembedding_len,numClasses,input_data, keep_prob,keep_prob2 )
    
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

    if not os.path.exists('ckpt/'):
        os.makedirs('ckpt/')
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(num_of_epochs):
            train_accuracy=[]
            train_cost=[]
            for i in range(0,len(train_data),batch_size):
                x_batch, y_true_batch = get_next_batch(batch_size, train_data,train_labels)
                _,trainig_batch_cost,training_batch_acc= sess.run([optimizer,loss,accuracy] , feed_dict = {input_data:x_batch, labels: y_true_batch, keep_prob:0.8, keep_prob2:0.8 })

                train_cost.append(trainig_batch_cost)
                train_accuracy.append(training_batch_acc)

            train_accuracy = np.array(train_accuracy)
            val_cost,val_acc= sess.run([loss,accuracy] , feed_dict = {input_data:val_data , labels: val_labels , keep_prob:1,keep_prob2:1})

            print("epoch: {} , Training accuracy:: {} , Validation accuracy:: {} ".format(epoch, train_accuracy.mean() , val_acc)) 

            if epoch %5==0 and epoch !=0:
                saver = tf.train.Saver()
                saver.save(sess,'./ckpt/checkpoint_'+str(epoch)+'.ckpt')
                tf.train.write_graph(sess.graph.as_graph_def(), './ckpt/', 'checkpoint_'+str(epoch)+'.pbtxt', as_text=True)
    
    
