import numpy as np
import tensorflow as tf
import re
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import csv
import numpy as np



def clean_text(text):
	    text = text.lower()
	    text = re.sub(r"i'm", "i am", text)
	    text = re.sub(r"he's", "he is", text)
	    text = re.sub(r"she's", "she is", text)
	    text = re.sub(r"that's", "that is", text)
	    text = re.sub(r"what's", "what is", text)
	    text = re.sub(r"where's", "where is", text)
	    text = re.sub(r"\'ll", " will", text)
	    text = re.sub(r"\'ve", " have", text)
	    text = re.sub(r"\'re", " are", text)
	    text = re.sub(r"\'d", " would", text)
	    text = re.sub(r"won't", "will not", text)
	    text = re.sub(r"can't", "cannot", text)
	    text = re.sub(r"won't", "will not", text)
	    text = re.sub(r"'s", "is", text) 
	    text = re.sub(r"[-()\"#/@;:<>${}+=~'|.?,!&%^*]", "", text)
	    return text
    
def encoding_question(text):
    
    text = clean_text(text)
    
    ques_id = np.load('ques_id.npy').item()
    
    encoded_ques = []
    
    for word in text.split():
         encoded_ques.append(ques_id[word])
    
    Rem = 55 - len(encoded_ques)
    
    for i in range(0,Rem):
        encoded_ques.append(0)
    
    return encoded_ques
        

def encoding_answer(id):
    
    id_answers = np.load('id_answers.npy').item()
    
    return id_answers[id]

    