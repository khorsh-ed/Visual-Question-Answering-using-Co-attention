import numpy as np
import keras
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
from model import question_hierarchy,parallel_co_attention
import pandas as pd
from Reading_Images_Features  import fun
from Reading_Testing_Images_Features  import Testfun
from tkinter import *
import pandas as pd
from tkinter import messagebox

dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\y_true_train.csv' , header=None)
y_true_train = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\Q_train.csv' , header=None)
Q_train = dataset.iloc[:, :].values
V_train=fun()

max_length = 55
vocab_size = 1000
num_class = 10

Q = Input(shape=(55,))
V = Input(shape=(512,49))
  
q_model = Sequential()
q_model.add(Embedding ( vocab_size , 512 , input_length=max_length))
q_model.add(LSTM(units=512, return_sequences=True, input_shape=(max_length, 512)))
q_model.add(Dropout(0.5))
q_model.add(LSTM(units=512, return_sequences=False))
q_model.add(Dense(1024, activation='tanh'))
encoded_q = q_model(Q)

V_model = Sequential()
V_model.add(Flatten(input_shape=(512,49)))
V_model.add(Dense(1024 ,activation='tanh'))
encoded_v = V_model(V)


Mul = keras.layers.multiply([encoded_q,encoded_v])


vqa_model=Dense(512, activation='tanh')(Mul)
vqa_model=Dropout(0.5)(vqa_model)
vqa_model=Dense(10, activation='softmax')(vqa_model)

model = Model(inputs=[Q,V], outputs=vqa_model)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([Q_train,V_train], [y_true_train],epochs=256, batch_size=300)

k=0
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\y_pred_test.csv' , header=None)
y_true_test = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\Q_test.csv' , header=None)
Q_test = dataset.iloc[:, :].values
V_test=Testfun()
pred = model.predict([Q_test,V_test],batch_size=300)
y_pred = np.full((len(pred[:,0]),len(pred[0,:])),0)
for i in range(len(pred[:,0])):
    mx = -1000
    idx =0
    for j in range(len(pred[0,:])):
        if(pred[i,j]>mx):
            mx=pred[i,j]
            idx=j
    y_pred[i,idx]=1
#loss, acc = model.evaluate(y_true_test, y_pred)
cnt = 0
for i in range(len(y_pred[:,0])):
    if np.all(y_pred[i,:] == y_true_test[i,:]):
        cnt = cnt + 1
        
ACC = (cnt/len(y_pred[:,0])) * 100
'''
def Predict():
    global k
    QA = y_true_test[k:k+1,:]
    QFe = Q_test[k:k+1,:]
    #for i in range(55):
    #    QFe[i,0] = Q_test[k,i]
    from PIL import Image 
    Image.open(str(LearningRate_Entry.get()) + ".jpg").show()
    dataset = pd.read_csv(str(LearningRate_Entry.get()) + ".csv" , header = None)
    x = dataset.iloc[:, 0:3].values
    ReadedV = np.full((1,512,49),0.0)
    for i in range(len(x[:,0])):
        ReadedV[0,np.int32(x[i,1]),np.int32(x[i,2])] = x[i,0]
    pred = model.predict([QFe,ReadedV],batch_size=1)
    t=0
    p=0
    mx = -1000
    for i in range(len(pred[0,:])):
        if pred[0,i] > mx:
            p = i;
            mx = pred[0,i]
        if QA[0,i] == 1:
            t = i;
            
    classes = []
    with open('E:\\Prototype Dataset\\Training\\classes.txt') as f:
        classes = f.readlines()
    classes = [x.strip() for x in classes] 
    if p == t:
        print("Info", "Correct Prediction\n\n","Answer: ",classes[t],"\tPrediction: ",classes[p])
    else:
        print("Info", "Wrong Prediction\n\n","Answer: ",classes[t],"\tPrediction: ",classes[p])
    k = k + 1

from tkinter import *

#Creating the main window
root = Tk()
#Controls

LearningRate_Label = Label(root , text = "Image Name")
LearningRate_Entry = Entry(root)
epochs_Label = Label(root , text = "Question")
epochs_Entry = Entry(root)
Train_Button = Button(root , text = "Predict the Answer" , command = Predict)
#Controls' positions
LearningRate_Label.grid(row=0 , column=0 )
LearningRate_Entry.grid(row=0 , column=1 )
epochs_Label.grid(row=1 , column=0 )
epochs_Entry.grid(row=1 , column=1 )
Train_Button.grid(row=2,column=1)
#For Making the window still displayed
root.mainloop()
'''

