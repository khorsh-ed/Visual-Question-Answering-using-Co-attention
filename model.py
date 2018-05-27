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

        
def question_hierarchy(Ques,max_length,vocab_size):

 word_level = Embedding ( vocab_size , 512 , input_length=max_length)(Ques)
 
 uni_gram = Conv1D( 512 , kernel_size = 1 , activation='tanh' , padding='same' )(word_level)
 bi_gram  = Conv1D( 512 , kernel_size = 2 , activation='tanh' , padding='same' )(word_level)
 tri_gram = Conv1D( 512 , kernel_size = 3 , activation='tanh' , padding='same' )(word_level)

 phrases = []
 
 for w in range(0,max_length):
    
   Uni =uni_gram [:,w,:]
   Uni= tf.reshape(Uni , [-1,1,512 ])
  
   Bi = bi_gram [:,w,:]
   Bi = tf.reshape(Bi , [-1,1,512 ])

   Tri= tri_gram [:,w,:]
   Tri = tf.reshape( Tri , [-1,1,512 ])

   Uni_Bi_Tri  =  Concatenate(axis=1)([ Uni, Bi , Tri ])
   Best_phrase = GlobalMaxPooling1D()(Uni_Bi_Tri)
   Best_phrase = tf.reshape( Best_phrase , [-1,1,512 ])
   phrases.append(Best_phrase)
  

 phrase_level = Concatenate(axis=1)(phrases)
 
 Question_level =  LSTM( 512  ,input_shape=(55,512) ,return_sequences='true')(phrase_level)
 
 return word_level, phrase_level, Question_level

def parallel_co_attention ( Q , V ):
      
 initializer = tf.contrib.layers.xavier_initializer()
 Wb = tf.Variable(initializer([512,512])) #shape(512,512)
 Q_Wb = K.dot(Q, Wb)  #(?,15,512)*(512,512) = (?,15,512)
 C = tf.nn.tanh(K.batch_dot(Q_Wb,V))  #(?,15,512)*(?,512,196) = (?,15,196)
 print("C = " , C.shape)
 """Computing WvV """
 
 Wvt = tf.Variable(initializer([512,512]))
 Vt = tf.transpose(V, perm=[0, 2, 1])  # V transposed
 Vt_Wvt = K.dot( Vt , Wvt ) #shape 
 Wv_V = tf.transpose(Vt_Wvt, perm=[0, 2, 1]) 

 print("WvV = ",Wv_V.shape)

 """WqQ_C """

 Wqt = tf.Variable(initializer([512,512]))
 Qt_Wvt = K.dot( Q , Wqt ) 
 Wq_Q = tf.transpose(Qt_Wvt , perm=[0, 2, 1])
 print("Wq_Q = ",Wq_Q.shape)
 Wq_Q_C = K.batch_dot(Wq_Q ,C)
 print("Wq_Q_C = ",Wv_V.shape)

 """Hv """
 Hv = tf.add(Wv_V,Wq_Q_C)
 Hv = tf.nn.tanh(Hv)
 print("Hv = ",Hv.shape)

 """av """

 Whv = tf.Variable(initializer([512,1]))
 Hvt = tf.transpose(Hv , perm=[0, 2, 1])

 av = tf.transpose(K.dot(Hvt , Whv), perm=[0, 2, 1])
 av = tf.nn.softmax(av)
 print("av = ",av.shape)

 v = K.batch_dot(av,Vt)
 print("v^ = ",v.shape)

 """attenstion of Question """



 """computng Hq """
 Ct = tf.transpose(C, perm=[0, 2, 1]) #transpose of C
 Hq = Wq_Q + K.batch_dot(Wv_V,Ct) 
 print("Hq = ",Hq.shape)

 
 """computng aq """
 Whq = tf.Variable(initializer([512,1])) #shape(k,1)
 Hqt = tf.transpose(Hq , perm=[0, 2, 1])

 aq = tf.transpose(K.dot(Hqt , Whq), perm=[0, 2, 1])
 aq = tf.nn.softmax(aq)
 print("aq = ",aq.shape)

 q = K.batch_dot(aq,Q)
 print("q^ = ",q.shape)
 return v,q

