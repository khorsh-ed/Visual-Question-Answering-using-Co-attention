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
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from Coattention import *
from Question_Hierarichy import *
import pandas as pd
from Reading_Images_Features  import LoadFeatures
from tkinter import *
import pandas as pd
from tkinter import messagebox


"""train"""
dataset=pd.read_csv('E:\\PreProcessed Text\\y_true_train.csv' , header=None)
y_true_train = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\PreProcessed Text\\Q_train.csv' , header=None)
Q_train = dataset.iloc[:, :].values
V_train=LoadFeatures("E:\\COCO-QA Dataset\\train\\img_ids.txt","E:\\COCO-QA Dataset\\Training Images Features\\CO_train2014_")

"""test"""
dataset=pd.read_csv('E:\\PreProcessed Text\\y_true_test.csv' , header=None)
y_true_test = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\PreProcessed Text\\Q_test.csv' , header=None)
Q_test = dataset.iloc[:, :].values
V_test=LoadFeatures("E:\\COCO-QA Dataset\\test\\img_ids.txt","E:\\COCO-QA Dataset\\Testing Images Features\\CO_val2014_") 


Q = Input(shape=(55,))
V = Input(shape=(512,49))

w_level ,p_level, q_level = Ques_Hierarchy(Q,V)

qw,vw = Co_attention([w_level,V])
qp,vp = Co_attention([w_level,V])
qs,vs = Co_attention([q_level,V])

w_att = keras.layers.Add()([qw,vw])
hw = Dense(512,activation='tanh')(w_att)
hw = Dropout(0.5)(hw)

p_att = keras.layers.Add()([qp,vp])
hp = keras.layers.concatenate([p_att,hw],axis=-1)
hp = Dense(512,activation='tanh')(p_att)
hp = Dropout(0.5)(hp)

s_att = keras.layers.Add()([qs,vs])
hs = keras.layers.concatenate([s_att,hp],axis=-1)

hs = Dense(512,activation='tanh')(s_att)
hs = Reshape((512,),input_shape=(1,512))(hs)
hs = Dropout(0.5)(hs)
p =  Dense(430,activation='softmax')(hs)

print(p.shape)


Rms = keras.optimizers.RMSprop(lr=0.0004, rho=0.9, epsilon=None, decay=0.00000001)
model = Model(inputs=[Q,V], outputs=p)
model.compile(optimizer=Rms, loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([Q_train,V_train], [y_true_train],epochs=256, batch_size=300,validation_data = ([Q_test,V_test],y_true_test ))

# save model as json file
saved_model = model.to_json()
with open("model.json", "w") as file:
    file.write(saved_model)

# save weights as hdf5
model.save_weights("model.h5")
print("Model saved successfully to disk")

# Loading model
file = open("model.json", "r")
loaded_model_json = file.read()
file.close()
loaded_model = model_from_json(loaded_model_json)

# Loading weights
loaded_model.load_weights("model.h5")
print("Loaded model successfully from disk")

