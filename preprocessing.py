import numpy as np
import tensorflow as tf
import re
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import csv


Mytypes = ['1','2','3']
Questions = open('E:\\Prototype Dataset\\Training\\questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
Answers = open('E:\\Prototype Dataset\\Training\\answers.txt').read().split('\n')
Types = open('E:\\Prototype Dataset\\Training\\types.txt').read().split('\n')

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
    


clean_questions = []

ps = PorterStemmer()
st_words= {}

for i in range(0 , len(Questions)):
    if(Types[i] in Mytypes):
        clean_questions.append(clean_text(Questions[i]))
"""       
for question in clean_questions:
    for word in question.split():
        st_words[word]=ps.stem(word)
       
"""  
word_freq ={}
for question in clean_questions:
   for word in question.split():
      #word = ps.stem(word)
      if word not in word_freq:
        word_freq[word]=1
      else:
        word_freq[word]+=1
           
Topwords = sorted([(count,word) for word,count in word_freq.items()], reverse=True)

Vocab = [word for count,word in Topwords]

Vocab = Vocab[:998]

Vocab.append('UNK')

ques_id = {Vocab[i]:i+1 for i in range(0,len(Vocab))} 

ques_id['UNK']=999

""" encoding questions """

encoding_questions=[]

for question in clean_questions:
   encode_question=[]
   for word in question.split():
       #word = ps.stem(word)
       if(word in ques_id):
           encode_question.append(ques_id[word])
       else:
           encode_question.append(ques_id['UNK'])
   encoding_questions.append(encode_question)

encoding_questions = pad_sequences(encoding_questions, maxlen=55, padding='post')
with open("E:\\Prototype Dataset\\PreProcessed Text\\Q_train" + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(encoding_questions)
""" Generating Answers """

encoding_answers = []
id_answers = {}
answers_id = {}
answer_freq = {}
answer_id = 0

for i in range(0,len(Answers)):
     if(Types[i] in Mytypes):
         if(Answers[i] not in answer_freq):
           answer_freq[Answers[i]]=1
         else:
           answer_freq[Answers[i]]+=1
 

for word in answer_freq :
    answers_id[word]= answer_id
    id_answers[answer_id]= word
    answer_id+=1

y_true = []

for i in range(0,len(Answers)):
     if(Types[i] in Mytypes):
         y_true.append(answers_id[Answers[i]])
         
y_true = np_utils.to_categorical(y_true, 10)
with open("E:\\Prototype Dataset\\PreProcessed Text\\y_true_train" + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(y_true)