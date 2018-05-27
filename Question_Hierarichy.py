import numpy as np
import keras 
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

def ComputePhrase(word_level):
    uni_gram = Conv1D( 512 , kernel_size = 1 , activation='tanh' , padding='same' )(word_level)
    bi_gram  = Conv1D( 512 , kernel_size = 2 , activation='tanh' , padding='same' )(word_level)
    tri_gram = Conv1D( 512 , kernel_size = 3 , activation='tanh' , padding='same' )(word_level)
    
    phrases = []
    
    for w in range(0,55):
        
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
    return Concatenate(axis=1)(phrases)
    

def Ques_Hierarchy(Ques , Img , max_length=55 ):
    
    word_level = Embedding ( 2000 , 512 , input_length=max_length)(Ques)

    phrase_level = Lambda(ComputePhrase)(word_level)
    
    Question_level =  LSTM(512,input_shape=(15,512) ,return_sequences='true')(phrase_level)
    #Question_level =  LSTM(512,input_shape=(15,512) ,return_sequences='true')(Question_level)
    return word_level ,phrase_level, Question_level









