
'''Packages for file processing'''
import os
import glob
import sys
'''Packages for data processing'''
import numpy as np
import pandas as pd
'''Packages to read PCAPs'''
from scapy.all import *
'''For padding'''
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelBinarizer


from sklearn.model_selection import train_test_split

import tensorflow as tf

'''Generate Raw Bytes'''
def generateRawBytes(pcap_path, out_csv, out_dir="data/raw_bytes"):
    """Convert a pcap capture into the byte-level CSV consumed by train.py.
    Example: generateRawBytes("4SICS-GeekLounge-151022.pcap",
                              "raw_bytes_4SICS-GeekLounge-151022.csv")"""
    scapy_cap = rdpcap(pcap_path)

    '''To create (src_ip, binary_payload) csv'''
    src_ip=[]
    bytes_raw=[]
    for pkt in scapy_cap:
        if IP in pkt:
            #src_ip.append(pkt[IP].src)#SAI- Uncomment if headers need to be included
            #bytes_raw.append(raw(pkt))#SAI- Uncomment if headers need to be included
            '''Following IF block for just payloads as inputs'''
            if Raw in pkt: #See: https://mpostument.medium.com/packet-sniffer-with-scapy-part-3-a895ce7e9cb
                src_ip.append(pkt[IP].src)
                bytes_raw.append(pkt['Raw'].load)
    df = pd.DataFrame({'src_ip':src_ip,'bytes':bytes_raw})
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(os.path.join(out_dir, out_csv))

# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]

# def split_dataset(X, y, size=0.3, shuffle_a = True):
#     '''See: https://stackoverflow.com/questions/70374217/how-to-get-validation-set-which-has-equal-number-of-images-for-each-class-using'''
#     ax = []
#     bx = []
#     ay = []
#     by = []
#     # Iterate over the labels
#     for label in np.unique(y):
#         count = 0
#         # Calculate the maximum number of values to include
#         max_count = size
#         for i in range(0,len(y)):
#             if y[i] == label: # Make sure we treat only a specific label
#                 if (count < max_count):
#                     ax.append(X[i])
#                     ay.append(y[i])
#                 else:
#                     bx.append(X[i])
#                     by.append(y[i])
#                 count += 1
                        
#     if shuffle_a:
#         ax, ay = unison_shuffled_copies(np.array(ax), np.array(ay))
#     else:
#         ax, ay = (np.array(ax), np.array(ay))
#     bx, by = unison_shuffled_copies(np.array(bx), np.array(by))
#     return ax, bx, ay, by




def my_train_test_splitV0(data, val_size=0.1):
    label = data[1].values
    bytes = data[2].values
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    split = int(len(data)*val_size)
    x_train, x_test = bytes[idx[split:]], bytes[idx[:split]]#original
    y_train, y_test = label[idx[split:]], label[idx[:split]]#original


    # label_as_binary = LabelBinarizer()
    # y_train = label_as_binary.fit_transform(y_train)
    # y_test = label_as_binary.fit_transform(y_test)
    return x_train, x_test, y_train, y_test

def my_train_test_split(data, train_ratio=0.8,test_ratio=0.1):

    #print(len(data))
    '''Remove classes with only one sample. See: https://stackoverflow.com/questions/43179429/scikit-learn-error-the-least-populated-class-in-y-has-only-1-member'''
    all_keys = data[1].unique().tolist()
    #print(len(all_keys))
    for key in all_keys:
        if data.loc[data[1]==key].shape[0] <=2 :#sai- '<'
            i = data.index[(data[1] == key)]
            #print(i)
            data.drop(i,inplace=True)
    #print(len(data))
    label = data[1].values
    #label = np.unique(label,return_inverse=True)[1]#sai- this is for missing/dropped label (see lines 101-105). Otherwise, it creates issues while conformalziing
    bytes = data[2].values

    #label_as_binary = LabelBinarizer()
    #labels_ohe = label_as_binary.fit_transform(label)
    #print(len(labels_ohe[0]))

    #x_train, x_test, y_train, y_test = train_test_split(bytes, label, stratify=label, train_size=train_ratio,test_size=test_ratio)
    x_train, x_test, y_train, y_test = train_test_split(
        bytes, label, stratify=label,
        train_size=train_ratio, test_size=test_ratio,
        random_state=4
    )
    y_train, y_test = binarizeLabels(y_train, y_test)
    return x_train, x_test, y_train, y_test    

'''One Hot Encode the labels after train_test_Split to save time'''
def binarizeLabels(y_train, y_test):
    y_train_len = len(y_train)
    y_test_len = len(y_test)

    merged_y = np.concatenate((y_train, y_test), axis=0)

    labels_as_binary = LabelBinarizer()
    labels_ohe = labels_as_binary.fit_transform(merged_y)

    y_train_ohe = labels_ohe[:y_train_len]
    y_test_ohe = labels_ohe[y_train_len:]

    return y_train_ohe, y_test_ohe


def preprocessV0(input_raw, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    IP_list = input_raw[1].values
    unique_ips = np.unique(IP_list)

    counter =0
    d = pd.DataFrame()
    corpus = []
    labels = []
    for ip in IP_list:
        bytes_list =  input_raw[2].loc[input_raw[1] == ip].to_list()
        #print(type(bytes_list[0]))
        #print(bytes_list[0:100])
        all_bytes = ''.join(map(str, bytes_list))
        #print(type(all_bytes[0]))
        #print(all_bytes[0:100])  
        all_bytes = str.encode(all_bytes)
        all_bytes = [x for x in all_bytes]
        #print(type(all_bytes[0]))
        #print(all_bytes[0:100])     
        corpus.append(all_bytes)
        labels.append(ip)
        #counter=counter+1
        #if(counter==2):
        #    break
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, labels #TODO: check if there is one-one mapping between bytes and labels (IPs)




def preprocessV1(input_raw, IP_labels,max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    unique_ips = np.unique(IP_labels)
    counter =0
    d = pd.DataFrame()
    corpus = []
    labels = []

    my_dict = dict()
    for key,value in zip(IP_labels,input_raw):#See: https://stackoverflow.com/questions/67846445/how-to-create-a-dictionary-with-multiple-values-per-key
        my_dict.update({key: my_dict.get(key, [])+[value]})
    for i in unique_ips:
        #print(i)
        bytes_list = my_dict.get(i)
        # print(type(bytes_list))
        # print(bytes_list)
        all_bytes = ''.join(map(str, bytes_list))
        #print(type(all_bytes[0]))
        #print(all_bytes[0:100])  
        all_bytes = str.encode(all_bytes)
        all_bytes = [x for x in all_bytes]
        corpus.append(all_bytes)
        labels.append(i)
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, labels #TODO: check if there is one-one mapping between bytes and labels (IPs)



'''
output must be compatible with keras.pad_Sequences and train_test_split
(1). Read the string formatted bytes for each IP and join them into a single string
(2). Encode the long string from (1) so it is in bytes form (again)
(3). Iterate over the list created from (2) so that each byte is decimal (decimal is needed for pad_sequences). 
(4) Append this list of decimals to corpus
(5) Now pad_sequences can be called on the corpus
Note: Corpus is a list of lists, where each individual list is a byte array (where each byte is a decimal)
'''
def preprocess(input_raw, IP_labels,max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    unique_ips = np.unique(IP_labels)
    counter =0
    d = pd.DataFrame()
    corpus = []
    labels = []

    # my_dict = dict()
    # for key,value in zip(IP_labels,input_raw):#See: https://stackoverflow.com/questions/67846445/how-to-create-a-dictionary-with-multiple-values-per-key
    #     my_dict.update({key: my_dict.get(key, [])+[value]})
    for i in range(len(input_raw)):
        #print(i)
        bytes_list = input_raw[i]
        # print(type(bytes_list))
        # print(bytes_list)
        all_bytes = ''.join(map(str, bytes_list))
        #print(type(all_bytes[0]))
        #print(all_bytes[0:100])  
        all_bytes = str.encode(all_bytes)
        all_bytes = [x for x in all_bytes]
        corpus.append(all_bytes)
        labels.append(IP_labels[i])
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, labels #TODO: check if there is one-one mapping between bytes and labels (IPs)





def data_generator(data, labels, train_flag, max_len=1000, batch_size=64, shuffle=True):
    idx = np.arange(len(data))
    if shuffle:
        np.random.seed(4)#for post-prediction analysis between y_Truth and y_pred
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)] #sends data to preprocess in batches of size 64. See: https://stackoverflow.com/questions/77293344/how-does-batch-size-in-this-code-control-the-number-of-output-results
    counter = 0
    while True:
        for i in batches:
                counter = counter+1
                xx = preprocess(data[i],labels[i], max_len)[0]#we return labels as well from preprocess() but can just stick to using the function args 'labels' as well
                #yy = np.asarray(yy)
                #print(yy[0])
                #print(type(labels_ohe))
                yy = labels[i]
                #print(xx.shape)
                #print(len(yy))
                #print(yy.shape)
                #print(xx[1].shape)
                #print(yy[1].shape)
                yield (xx, yy)
        if train_flag==False:
            break

