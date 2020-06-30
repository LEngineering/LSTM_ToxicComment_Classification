# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:13:50 2020

@author: crist
"""

import pandas as pd
import numpy as np
#from textblob import TextBlob
from pandas import DataFrame
import os,sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt

#Define the train path and split into input and output
files_path = "/Users/crist/Documents/PYTHON/Kagglerec/"
train_dr=pd.read_csv("jigsaw-toxic-comment-train.csv")


#Data pre-processing 
#function to remove punctuation
def remove_punct(text):
    import string
    transf=str.maketrans('','', string.punctuation)
    return text.translate(transf)
train_dr['comment_text']=train_dr['comment_text'].apply(remove_punct)

stopw = stopwords.words('english')
#function for removing the stop words
def removestopw(txt):
    txt=[word.lower() for word in txt.split() if word.lower() not in stopw] #removing the stopwords and lowercasing
    return " ".join(txt) #joining the list of words with " "

train_dr['comment_text']=train_dr['comment_text'].apply(removestopw)

#stemming words (eq: reverted->revert)
stemmer=SnowballStemmer("english")
def stemword(txt):
    txt=[stemmer.stem(word) for word in txt.split()]
    return " ".join(txt)
train_dr['comment_text']=train_dr['comment_text'].apply(stemword)

###Split X and Y
train_X=np.array(train_dr['comment_text'])
train_Y=np.array(train_dr['toxic'])

# Preparing data 
# Convert text data into token vectors
t= Tokenizer()

#Train the tokenizer to the text
t.fit_on_texts(list(train_X)) # it fits each word to a number in train_X
#Convert list of strings into a list of integers
seq_train = t.texts_to_sequences(train_X)
#seq_train[6][:] #[6][:] 


#Mapping of indexes to words 
word_index = t.word_index # a dictionary of words and their uniquely assigned integers.
print(' %s unique assigned integers.' % len(word_index))
vocab_size=len(word_index)

#maxlenght of strings train
maxlen=max((max(len(seq) for seq in seq_train),
            ))

#Padding to calculate the max size of the train sequences
#this is X for train
Pad_tr=sequence.pad_sequences(seq_train,maxlen=maxlen,dtype='int32',padding='post',
                     truncating='post',value=0)

#split the data into training (80%), validation (10%) and testing (10%)
#split to train, test
(X_train,X_test, Y_train,Y_test) = train_test_split(Pad_tr, train_Y,
	test_size=0.10, random_state=1)
#split again train into validation and train 
(X_train,X_val, Y_train,Y_val) = train_test_split(X_train,Y_train,
	test_size=0.10, random_state=1)

#===============================GENERATE EMBEDDINGS = ONLY RUN ONCE = START ============================================

dim_embedding=300

#Get embedding
idx_embedding={} #new embedding dictionary
embedding_file=open('wiki.en.vec', encoding='utf8') #use wiki en vec to assign word vectors
for row in embedding_file:
    val=row.rstrip().rsplit(' ',dim_embedding) #split a string into a list and rest embedding coefficients and next removes spaces
    word=val[0] #selects word
    coef=np.asarray(val[1:],dtype='float32') #selects embedding coefficients
    idx_embedding[word] = coef #transforms to idx_embedding dictionary
embedding_file.close() #closes wiki en vec file 



#===============================GENERATE EMBEDDINGS = ONLY RUN ONCE = END ============================================

#Creat embedding matrix
matrix_embedding=np.zeros((len(word_index)+1,dim_embedding)) #initialises the embinedding matrix for words in training set 
for word,j in word_index.items(): #for each word "word" and index "j" in word_index dictionary items generated from training set
    vec_embed=idx_embedding.get(word) #defines an embedding vector for each word taken form training set in embedding dictionary 
    if vec_embed is not None: #if there's an embedding (not none)
        matrix_embedding[j]=vec_embed #attach embedding

#Save embeddings
import h5py
with h5py.File('embeddingm.h5','w') as hf: #creates new h5py file  
    hf.create_dataset('fasttext', data=matrix_embedding) #saves embedding matrix to h5 file
    
#Load embeddings
with h5py.File('embeddingm.h5','r') as hf:
    matrix_embedding=hf['fasttext'][:]
    

#Model
import keras.backend
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D,LSTM
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

#Initial model
model=Sequential()

#Add embedding layers 
model.add(Embedding(vocab_size+1,dim_embedding, weights=[matrix_embedding], 
                    input_length=maxlen,trainable=False)) #set trainable false to keep embedding fixed?


model.add(LSTM(70, return_sequences=True,dropout=0.2, activation='tanh',inner_activation='sigmoid',
                name='lstm_layer')) 

model.add(Conv1D(filters=100, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())

# Add fully connected layers
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))#val
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))#val
model.add(Dense(1, activation='sigmoid')) 
# Summarize the model
model.summary()

# Set variables
lr=0.0005 
batch_size = 32 
epochs = 5

# compile the model
print("----------Compiling model----------")
               
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping
estop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

graph = model.fit(X_train, Y_train,epochs=epochs,
                  validation_data=(X_val, Y_val),shuffle=False,verbose=1,callbacks=[estop])
                 
model.save('toxiccom.h5')


###################################################################

#Generate graphs
loss = graph.history['loss']
val_loss = graph.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_valtrain.png')

accuracy = graph.history['accuracy']
val_accuracy = graph.history['val_accuracy']
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
plt.savefig('acc_valtrain.png')

loss, accuracy = model.evaluate(X_train, Y_train,verbose=1)
print('Accuracy_train: %f' % (accuracy*100))
loss_val, accuracy_val= model.evaluate(X_val, Y_val,verbose=1)
print('Accuracy_val: %f' % (accuracy_val*100))

loss_test, accuracy_test= model.evaluate(X_test, Y_test,verbose=1)
print('Accuracy_test: %f' % (accuracy_test*100))

#Predict test data
predict = model.predict(X_test, verbose=0)








