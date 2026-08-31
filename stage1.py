'''Packages for data processing'''
import numpy as np
import pandas as pd



'''Packages for file processing'''
import os
import glob
import sys



'''Packages for ML'''
import tensorflow as tf
import keras as kr
from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation, Reshape

from numpy.random import seed

'''Set random seeds using numpy and tensorflow because Keras uses both for background processes'''
seed(2)
tf.random.set_seed(2)


'''Hyper params'''
#num_samples=10000
# batch_size=1#32 yeilds best results
# epochs=10


# max_features = 22576#10000#2399
# sequence_length = 250#250
# embedding_dim = 128

# def Malconv(max_len=200000, win_size=500, vocab_size=256):    
#     inp = Input((max_len,))
#     emb = Embedding(vocab_size, 8)(inp)

#     conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
#     conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
#     a = Activation('sigmoid', name='sigmoid')(conv2)
    
#     mul = multiply([conv1, a])
#     a = Activation('relu', name='relu')(mul)
#     p = GlobalMaxPool1D()(a)
#     d = Dense(64)(p)
#     out = Dense(1, activation='sigmoid')(d)

#     return Model(inp, out)


# def Stage1(n_outputs,max_len=1000, win_size=500, vocab_size=256):  #correct model

#     #n_outputs =  64#TODO: why is it not 65?
#     #max_features = 22576#10000#2399
#     #sequence_length = 250#250
#     embedding_dim = 128

#     inp = Input((max_len,))
#     #emb = Embedding(max_features + 1, embedding_dim)(inp)
#     emb = Embedding(vocab_size, 8)(inp) #--->actual
#     #emb = Embedding(vocab_size,embedding_dim)(inp)

    

#     conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
#     conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
#     a = Activation('sigmoid', name='sigmoid')(conv2)
    
#     mul = multiply([conv1, a])
#     a = Activation('relu', name='relu')(mul)#--->actual
#     p = GlobalMaxPool1D()(a)
#     d = Dense(128)(p)
#     out = Dense(n_outputs, activation='softmax')(d)

#     return Model(inp, out)


def Stage1(n_outputs,max_len=1000, win_size=500, vocab_size=256):  #test model only. remove once finalized

    #n_outputs =  64#TODO: why is it not 65?
    #max_features = 22576#10000#2399
    #sequence_length = 250#250
    embedding_dim = 128

    inp = Input((max_len,))
    #emb = Embedding(max_features + 1, embedding_dim)(inp)
    emb = Embedding(vocab_size, 8)(inp) #--->actual
    #emb = Embedding(vocab_size,embedding_dim)(inp)

    

    #conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    #a = Activation('sigmoid', name='sigmoid')(conv2)#Option 1-> original
    a = Activation('relu', name='relu')(conv2)#Option 2
    
    #mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(conv2)#--->actual
    p = GlobalMaxPool1D()(a)
    d = Dense(128)(p)
    out = Dense(n_outputs, activation='softmax')(d)

    return Model(inp, out)


def Stage2(x_data):
    from sklearn.neighbors import LocalOutlierFactor
    from pyod.models.lof import LOF 

    contamination = 0.1  # percentage of outliers --> set it to top-K
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # train LOF detector
    clf_name = 'LOF'
    clf = LOF()
    clf.fit(x_data)
