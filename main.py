import matplotlib.pyplot as plt

import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Concatenate,Reshape , Input ,LSTM, Dense, Dropout ,concatenate , Flatten ,GlobalMaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D , Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation,Lambda
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from model import question_hierarchy,parallel_co_attention
import pandas as pd
from Reading_Images_Features  import fun
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def generate_batch(ques,y_true,image_feature,index):
    minimum = min(len(ques),index+300)  
    q=ques[index:minimum,:]
    y=y_true[index:minimum , :]
    v=image_feature[index:minimum,:]
    return q,y,v,index+300


vocab_size = 1000 
max_length = 55
batchsize = 300
Num_of_Classes = 10
NUM_OF_EPOCHS = 10
#dataset=pd.read_csv('E:\GP Downloads\y_true_test.csv' , header=None)
#y_true_test = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\y_true_train.csv' , header=None)
y_true_train = dataset.iloc[:, :].values
#dataset=pd.read_csv('E:\GP Downloads\Q_test.csv' , header=None)
#Q_test = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\Q_train.csv' , header=None)
Q_train = dataset.iloc[:, :].values
V_train=fun()

Ques = tf.placeholder(tf.float32, [None, 55])
V = tf.placeholder(tf.float32, [None, 512 , 49])
y_true = tf.placeholder(tf.float32, [None, Num_of_Classes])


word_level , phrase_level , question_level = question_hierarchy(Ques,max_length,vocab_size)
vw, qw = parallel_co_attention(word_level, V)
vp, qp = parallel_co_attention(phrase_level, V)
vs, qs = parallel_co_attention(question_level, V)


initializer = tf.contrib.layers.xavier_initializer()

Ww = tf.Variable(initializer([512,512])) 
bw = tf.Variable(initializer([1,512])) 
hw = tf.nn.dropout(K.dot(tf.add(qw,vw),Ww)+ bw , keep_prob = 0.5)

Wp  = tf.Variable(initializer([1024,512])) 
bp  = tf.Variable(initializer([1,512])) 
hp_ = tf.concat([tf.add(qp, vp), hw], axis=2)
hp  =  tf.nn.dropout(K.dot(hp_,Wp) + bp , keep_prob = 0.5)

Ws  = tf.Variable(initializer([1024,512])) 
bs  = tf.Variable(initializer([1,512])) 
hs_ = tf.concat([tf.add(qs, vs), hp], axis=2)
hs =  tf.nn.dropout( K.dot(hs_,Ws) + bs ,  keep_prob = 0.5)


Wh = tf.Variable(initializer([512,Num_of_Classes]))
bh = tf.Variable(initializer([1,Num_of_Classes]))
y_pred =  tf.nn.softmax(K.dot(hs,Wh) + bh)
y_pred = tf.reshape( y_pred , [-1,Num_of_Classes ])
print(y_pred.shape)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
ACC = []
Epoch = []
with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(NUM_OF_EPOCHS):
    
        #generate Batch 
        index=0
        while index<len(Q_train):
            Q_batch,y_batch,V_batch,index=generate_batch(Q_train,y_true_train,V_train,index)
           # print("Epoch ", i ," :  Acc = ")
            sess.run(train,feed_dict={Ques:Q_batch ,y_true:y_batch , V:V_batch})
        
        matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        Accuracy = sess.run(acc,feed_dict={Ques:Q_train ,y_true:y_true_train ,V:V_train})
        print("Epoch : ",i,"  Acc = ",Accuracy)
        ACC.append(Accuracy)
        Epoch.append(i)
    

ax1.clear()        
ax1.plot(Epoch,ACC)    
plt.show()









 




