import numpy as np
import keras 
import pandas as pd
from keras.models import Sequential,Model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Concatenate,Reshape , Input ,LSTM, Dense, Dropout ,concatenate , Flatten ,GlobalMaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D , Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation,Lambda
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from Question_Hierarichy import Ques_Hierarchy


def Transpose(tensor):
    return K.permute_dimensions(tensor,(0,2,1))

def Compute_c(x):
    h_level = x[0]
    V = x[1]
    Q_Wb = Dense(512,activation='tanh')(h_level)
    return K.tanh(K.batch_dot(Q_Wb,V))

def Compute_Hv(x):
    Wq_Q=x[0]
    C = x[1]
    Wv_v = x[2]
    return K.tanh(keras.layers.Add()([K.batch_dot(Wq_Q,C),Wv_v]))

def Soft_max(x):
    return K.softmax(x)

def Compute_Wv_V_Ct(x):
    Wv_v = x[0]
    Ct = x[1]
    return K.batch_dot(Wv_v, Ct )

def Compute_Hq(x):
    Wq_Q = x[0]
    Wv_V_Ct = x[1]
    return K.tanh(keras.layers.Add()([Wq_Q , Wv_V_Ct]))

def Compute_V_att(x):
    av =x[0]
    Vt =x[1]
    return  K.batch_dot(av,Vt)

def Compute_q_att(x):
     aq=x[0]
     h_level = x[1]
     return K.batch_dot(aq,h_level)



def Co_attention(x):
    
    h_level=x[0]
    V = x[1]
    k_dim = 512
    
    C =  Lambda(Compute_c)([h_level,V])
    Vt = Lambda(Transpose)(V)
    Wv_v = Dense( k_dim , activation='tanh' )(Vt)
    Wv_v = Lambda(Transpose)(Wv_v)
    print("Wv_v",Wv_v.shape)
    Wq_Q = Dense( k_dim , activation='tanh')(h_level)
    Wq_Q = Lambda(Transpose)(Wq_Q)
    print("Wq_Q",Wq_Q.shape)
    Hv = Lambda(Compute_Hv)([Wq_Q,C,Wv_v])
    Hvt= Lambda(Transpose)(Hv)
    av = Dense(1,activation='tanh')(Hvt) 
    av = Lambda(Transpose)(av)
    av = Lambda(Soft_max)(av)
    print(av.shape)
    Ct = Lambda(Transpose)(C)
    Wv_V_Ct = Lambda(Compute_Wv_V_Ct)([Wv_v,Ct])
    Hq = Lambda(Compute_Hq)( [Wq_Q , Wv_V_Ct])
    Hqt = Lambda(Transpose)(Hq)
    aq =  Dense(1,activation='tanh')(Hqt)
    aq =  Lambda(Transpose)(aq)
    aq = Lambda(Soft_max)(aq)
    
    v_att = Lambda(Compute_V_att)([av,Vt])
    q_att = Lambda(Compute_q_att)([aq,h_level])
    return q_att,v_att
