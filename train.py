from os.path import join
import argparse
import pickle
import warnings
import pandas as pd
import keras as kr
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

#import tensorflow_addons as tfa
import itertools

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, jaccard_score

from keras.callbacks import CSVLogger

#from sklearn.preprocessing import LabelBinarizer
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import utils
import stage1 as sg1

from numpy.random import seed
import numpy as np

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

'''Set random seeds using numpy and tensorflow because Keras uses both for background processes'''
seed(2)
tf.random.set_seed(2)





class BCP(kr.callbacks.Callback):
    batch_accuracy = [] # accuracy at given batch
    batch_loss = [] # loss at given batch    
    batch_auc = []
    def __init__(self):
        super(BCP,self).__init__() 
    def on_train_batch_end(self, batch, logs=None):                
        BCP.batch_accuracy.append(logs.get('categorical_accuracy'))
        BCP.batch_loss.append(logs.get('loss'))
        BCP.batch_auc.append(logs.get('auc'))

class EpochLogger(kr.callbacks.Callback):
    def __init__(self,**kargs):
        super(EpochLogger,self).__init__(**kargs)
        self.epoch_accuracy = {} # loss at given epoch
        self.epoch_loss = {} # accuracy at given epoch

    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch. 
        return

    def on_epoch_end(self, epoch, logs={}):
        # things done on end of the epoch
        self.epoch_accuracy[epoch] = logs.get("categorical_accuracy")
        self.epoch_loss[epoch] = logs.get("val_loss")
        self.model.save_weights("name-of-model-%d.h5" %epoch) # save the model


'''You can also get the truth labels for ROC Curve separately into a list y_true[]
    Useful: 
    [1] https://gist.github.com/RyanAkilos/3808c17f79e77c++4117de35aa68447045
    [2] https://stackoverflow.com/questions/37615544/f1-score-per-class-for-multi-class-classification
    [3] https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    [4] https://github.com/vinyluis/Articles/tree/main/ROC%20Curve%20and%20ROC%20AUC?source=post_page-----294fd4617e3a--------------------------------
    [5] https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
    
    Note: getting y_true is only needed for finer metric calculations like confusion matrix, AUC, ROC, F1-micro and F1-macro
    The metrics can be calculated per class and overall
    ''' 
'''Now you can plot ROC curve using y_true vs y_pred'''
def get_y_true(method=4):
    if method==1:
        '''Method 1'''
        test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
        (x_true, y_true) = list(enumerate(test_gen))
    if method==2:
        '''Method 2'''
        import more_itertools as mi
        test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
        (x_true, y_true) = list(mi.always_iterable(test_gen))
    if method==3:
        '''Method 3'''
        y_true = []
        for x,y in test_gen:
            y_test.append(y)
    if method==4:
        '''Method 4'''    
        test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
        y_true = [y for (x,y) in test_gen]
        test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
        x_true = [x for (x,y) in test_gen]
    if method==5:
        '''Method 5'''
        itt = iter(test_gen)
        #do the above 4 methods on itt
    if method==6:
        '''Method 6'''
        import pandas as pd
        test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
        df = pd.DataFrame(test_gen)


    '''The y_true[] is batched i.e. list of lists with each len(list)=64
    So they need to be joined together for enabling comparison with y_pred'''
    import itertools
    y_true = list(itertools.chain.from_iterable(y_true))
    '''Get the max of each individual output in y_true'''
    y_true = np.argmax(y_true, axis=1)

    '''Unbacth the x_true values as well i.e. from (64,1000) each into (1000,) each'''
    x_true = list(itertools.chain.from_iterable(x_true))

    return x_true, y_true

'''Confusion Matrix'''
def getConfusionMatrix(y_true,y_pred):
    #See: Confusion Matrix wont work over y_true vs y_pred because y_true is one-hot coded vectors
    #more information is here in the comments: https://stackoverflow.com/questions/70775762/how-to-make-a-confusion-matrix-with-keras
    #y_true = get_y_true()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred , normalize='pred')
    return cm

    


def train(model, max_len=1000, batch_size=64, verbose=True, epochs=500, save_path='../saved/', save_best=False):
    
    # callbacks
    csv_logger = CSVLogger("stage1_history_log.csv", append=True)
    ear = EarlyStopping(monitor='loss', patience=5)
    mcp = ModelCheckpoint(join(save_path, 'malconv.h5'), 
                          monitor="categorical_accuracy",
                          save_freq='epoch', 
                          save_best_only=save_best, 
                          save_weights_only=False)
    
    # history = model.fit(
    #     utils.data_generator(x_train, y_train, max_len, batch_size),
    #     steps_per_epoch=len(x_train)//batch_size + 1,
    #     epochs=epochs, 
    #     verbose=verbose, 
    #     callbacks=[ear, mcp,csv_logger,custom_check],
    #     validation_data=utils.data_generator(x_test, y_test, max_len, batch_size),
    #     validation_steps=len(x_test)//batch_size + 1)
    
    '''Test call'''
    history = model.fit(
        utils.data_generator(x_train, y_train, True, max_len, batch_size),
        steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
        epochs=epochs, 
        verbose=verbose, 
        #callbacks=[ear, mcp,csv_logger,epoch_logger,batch_logger],
        #validation_data=utils.data_generator(x_val, y_val, True, max_len, batch_size),
        validation_steps=int(np.ceil(len(x_val)/batch_size)))
    # print(custom_check.epoch_accuracy)
    return history


def train_val_results(BCP):
    
    '''Train. and Val. Results'''

    '''Batch-level Metrics'''
    plt.plot(BCP.batch_accuracy)
    plt.plot(BCP.batch_loss)
    plt.plot(BCP.batch_auc)
    plt.legend(['Accuracy','Loss','AUC'],loc='upper right')
    plt.show()


    '''Epoch-level Metrics'''
    print(len(history.history))
    print(history.history)

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['auc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Accuracy', 'Loss', 'AUC'], loc='upper right')
    plt.show() 


def testResults(y_true,y_pred):
        '''Testing Results'''

        '''Individual Metrics'''
        f1_score(y_true, y_pred, average='micro')
        f1_score(y_true, y_pred, average='macro')
        f1_score(y_true, y_pred, average='weighted')
        precision_score(y_true, y_pred,average='macro')
        precision_score(y_true, y_pred,average='micro')
        recall_score(y_true, y_pred,average='macro')
        recall_score(y_true, y_pred,average='micro')
        accuracy_score(y_true,y_pred)

        '''Classification Report'''
        report = classification_report(y_true, y_pred)#, target_names=class_labels)
        print(report)


        '''Confusion Matrix'''
        cm = getConfusionMatrix(y_true,y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)#, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

        '''Get Misclassified Samples'''
        sg1_misclassfied_samples = np.absolute(y_true-y_pred)
        #get indices of misclassified smaples from then retreive x_true

def conformal_train_predict(model, x_train,y_train,x_val, max_len=1000, batch_size=64, verbose=True, epochs=1, save_path='../saved/', save_best=False):
    # callbacks
    csv_logger = CSVLogger("stage1_history_log.csv", append=True)
    ear = EarlyStopping(monitor='loss', patience=5)
    mcp = ModelCheckpoint(join(save_path, 'malconv.h5'), 
                          monitor="categorical_accuracy",
                          save_freq='epoch', 
                          save_best_only=save_best, 
                          save_weights_only=False)
    history = model.fit(
        utils.data_generator(x_train, y_train, True, max_len, batch_size),
        steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
        epochs=epochs, 
        verbose=verbose, 
        callbacks=[ear, mcp,csv_logger,epoch_logger,batch_logger])#,
    #    validation_data=utils.data_generator(x_val, y_val, True, max_len, batch_size),
    #    validation_steps=int(np.ceil(len(x_val)/batch_size)))
    test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
    f = model.predict(test_gen)

    
    return f,model


def calculate_q_yhat_naive(model, x_train,y_train,x_val,y_val,alpha):
    
    f,model=conformal_train_predict(model, x_train,y_train,x_val)


    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    y_cal = [y for (x,y) in val_gen]
    import itertools
    y_cal = list(itertools.chain.from_iterable(y_cal))
    # y_cal = np.argmax(y_cal, axis=1)


    N=len(y_cal)
    q_yhat=np.quantile(np.abs(y_cal-f),np.ceil((N+1)*(1-alpha))/N)
    
    return q_yhat,model


def calculate_qyhat_naive_classification(softmax_outputs,x_val,y_val,alpha):
    N=softmax_outputs.shape[0]
    scores=np.zeros(N)

    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    y_cal = [y for (x,y) in val_gen]
    import itertools
    y_cal = list(itertools.chain.from_iterable(y_cal))
    y_cal = np.argmax(y_cal, axis=1)

    
    for i in range(N):
        true_softmax_output=softmax_outputs[i][y_cal[i]]
        scores[i]= 1-true_softmax_output
        
    q_yhat=np.quantile(scores,np.ceil((N+1)*(1-alpha))/N)
    
    return q_yhat


def Calculate_q_yhats(model,x_val, y_val, alpha=0.1):
    cal_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    import itertools
    x_cal = [x for (x,y) in cal_gen]
    x_cal = list(itertools.chain.from_iterable(x_cal))
    cal_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)

    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    y_val_n = [y for (x,y) in val_gen]
    
    y_val_n = list(itertools.chain.from_iterable(y_val_n))
    y_val_n = np.argmax(y_val_n, axis=1)
    y_val_n = y_val_n.tolist()


    y_pred = model.predict(cal_gen)
    print("len(y_pred[0]): "+str(len(y_pred[0])) )
    N=len(x_cal)#.shape[0]
    print("N: "+str(N))
    d_alpha={}
    d_alpha_q_yhats={}
    conf_pred=[]

    for i in range(N):
        real_class=y_val_n[i]
        prob=1-y_pred[i][real_class]
        
        if real_class not in d_alpha:
            d_alpha[real_class]=[prob]
        else:
            d_alpha[real_class].append(prob)

    for i,(k,v) in enumerate(d_alpha.items()):
        N=len(v)
        q_hat = np.ceil((N+1)*(1-alpha))/N
        if q_hat > 1:
            q_hat = 1
        d_alpha_q_yhats[k]=np.quantile(v,q_hat)
        
            
        
    return d_alpha_q_yhats

def calculate_coverage(lower_bound,upper_bound,y_true):
    out_of_bound=0
    N=len(y_true)
    
    for i in range(N):
        if y_true[i]<lower_bound[i] or y_true[i]>upper_bound[i]:
            out_of_bound+=1
            
    return 1-out_of_bound/N
     
def calculate_class_coverage_v1(conf_sets, y_true):
    if hasattr(y_true[0], "__len__"):
        y_true = np.argmax(y_true, axis=1)
    matches = [true_label in pred_set for true_label, pred_set in zip(y_true, conf_sets)]
    coverage = np.mean(matches)

    return coverage

def calculate_class_coverage(conf_pred,y_true):
    s=0
    if hasattr(y_true[0], "__len__"):
        y_true = np.argmax(y_true, axis=1)
    
    for i in range(len(conf_pred)):
        if y_true[i] in conf_pred[i]:
            s+=1
    
    return s/len(y_true)


def calculate_class_bal_coverage(conf_sets,y_true):
    if hasattr(y_true[0], "__len__"):
        y_true = np.argmax(y_true, axis=1)
    d1=np.zeros(n_outputs)
    d2=np.zeros(n_outputs)

    for i in range(len(conf_sets)):
        d2[y_true[i]]+=1
    
        if y_true[i] in conf_sets[i]:
            d1[y_true[i]]+=1

    class_coverages = d1/d2
    
    return class_coverages



#function to get predict sets
def get_confsets_naive(model,x_val,y_val,q_yhat):
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    softmax_outputs = model.predict(val_gen)
    #softmax_outputs = np.argmax(softmax_outputs, axis=1)
    N=softmax_outputs.shape[0]
    
    conf_sets=[]
    
    for i in range(N):
        aux=[]
        for j in range(softmax_outputs.shape[1]):
            if softmax_outputs[i][j]>= 1-q_yhat:
                aux.append(j)
        conf_sets.append(aux)
        
    return conf_sets



def ConformalPrediction_classbalanced(model,x_val, y_val, d_alpha_q_yhats,adv_attack=False):

    conf_sets=[]
    if not adv_attack:
        val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
        y_pred = model.predict(val_gen)
    else:
        y_pred = model.predict(x_val)
    
    for i in range(y_pred.shape[0]):
        conf=[]
        for j in range(y_pred.shape[1]):
            #print("j: "+str(j))
            if 1-y_pred[i][j]<d_alpha_q_yhats.get(j,0.0):
                conf.append(j)
        conf_sets.append(conf)
        
    return conf_sets


def generate_adv_set(adv_samples, y_test, x_train, y_train,percentage=0.3):
    train_gen = utils.data_generator(x_train, y_train, False, max_len=1000, batch_size=64,shuffle=True)
    y_train_n = [y for (x,y) in train_gen]
    y_train_n = list(itertools.chain.from_iterable(y_train_n))
    #y_train = np.argmax(y_true, axis=1)
    train_gen = utils.data_generator(x_train, y_train, False, max_len=1000, batch_size=64,shuffle=True)
    x_train_n = [x for (x,y) in train_gen]
    '''Unbacth the x_true values as well i.e. from (64,1000) each into (1000,) each'''
    x_train_n = list(itertools.chain.from_iterable(x_train_n))

    combined_len = len(adv_samples) + len(x_train_n)
    np.random.seed(4)
    import random
    adv_set_size = int(np.round(combined_len*percentage))
    norm_set_size = int(np.round(combined_len*(1-percentage)))
    '''If there is not a minimum 30% that can be taken from adv. samples'''
    if len(adv_samples) < adv_set_size or len(x_train_n) < norm_set_size:
        adv_set_x = np.vstack((adv_samples,x_train_n))
        adv_set_y = np.vstack((y_test,y_train_n))
    else:
        sampled_adv_samples = random.sample(adv_samples.tolist(),adv_set_size)
        sampled_x_train = random.sample(x_train_n,norm_set_size)
        adv_set_x = np.vstack((sampled_adv_samples,sampled_x_train))
        sampled_y_test = random.sample(y_test,adv_set_size)
        sampled_y_train = random.sample(y_train_n,norm_set_size)
        adv_set_y = np.vstack((sampled_y_test,sampled_y_train))

    return adv_set_x, adv_set_y

if __name__ == '__main__':

    # prepare data
    dirname = r"data/raw_bytes/."
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_4SICS-GeekLounge-151020.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_4SICS-GeekLounge-151021.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'payload_raw_bytes_4SICS-GeekLounge-151022.csv'), header=None,skiprows=[0])
    input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC60870_IDS.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_DM_AS1.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_DM_AS2.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_DM_AS_combined.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_DoS_AS1.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_MS_AS1.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_MS_AS2.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_MS_AS3.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_MS_AS4.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_IEC68150_MS_AS_combined.csv'), header=None,skiprows=[0])
    #input_raw = pd.read_csv(join(dirname,'raw_bytes_RICSel21-IEC104-DoS_104_Flood.csv'), header=None,skiprows=[0])




    x_train, x_val, y_train, y_val = utils.my_train_test_split(input_raw)#TEST: PASS!
    print('Train on %d data, test on %d data' % (len(x_train), len(x_val)))
    n_outputs = len(y_train[0])


    model = sg1.Stage1(n_outputs, max_len=1000, win_size=500)
    model.compile(
    loss='categorical_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=1e-5),
    metrics=[kr.metrics.categorical_accuracy, kr.metrics.AUC(name='auc')])#, kr.metrics.F1Score(name='f1_Score')])#, kr.metrics.Recall(), kr.metrics.Precision()])
    model.summary()

    epoch_logger = EpochLogger()
    batch_logger = BCP()

    '''Normal Training Step'''
    history = train(model,max_len=1000, batch_size=32,epochs=2)



    '''Evaluate or Predict'''
    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw)#TEST: PASS!
    
    np.array_equal(x_val,x_test)#x_test!=x_val i.e. validation and test sets are not the same! 
    
    test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
    #model.evaluate(test_gen,steps=len(y_test)/64,verbose=1)#evaluate (alternative to predict)
    y_pred = model.predict(test_gen)#,steps=len(y_test)/64,verbose=1) 
    y_pred = np.argmax(y_pred, axis=1)

    '''Get y_true value for comparison'''
    x_true, y_true = get_y_true()

    '''Compare Results'''
    testResults(y_true,y_pred)
    

    '''Adversarial Sample Generation with CW Algorithm'''
    from cw import CW
    #get 30% adv samples --> 30% test set size
    ADV_PER = 0.5
    VAL_PER = 0.1
    TRAIN_PER = 1.0-VAL_PER-ADV_PER
    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw,train_ratio=TRAIN_PER,test_ratio=ADV_PER)#TEST: PASS!
    x_true, y_true = get_y_true()

    atk = CW(model, x_train, y_train, x_val, y_val, x_test, y_test, c=1, kappa=0, steps=100, lr=0.01)
    atk.set_model_training_mode(model_training=True, batchnorm_training=False, dropout_training=False)
    adv_y_test = atk(x_true, y_true)

    adv_y_test = adv_y_test.detach().numpy()
    #adv_list = adv_y_test.tolist()

    yy=model.predict(adv_y_test)
    yy_pred = np.argmax(yy, axis=1)


    '''Compare results'''
    testResults(y_true,yy_pred)
    

    '''Uncertainty Quantification using Conformal Prediction'''
    '''See:https://github.com/Quilograma/ConformalPredictionTutorial/blob/main/Conformal%20Prediction.ipynb'''
    '''Iterative CP'''
    alpha=0.001
    q_yhat,model=calculate_q_yhat_naive(model,x_train,y_train,x_val,y_val,alpha)
    print(q_yhat)
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    f_val=model.predict(val_gen).flatten()

    y_val = np.argmax(y_val, axis=1)
    print('Coverage of {}'.format(calculate_coverage(f_val-q_yhat,f_val+q_yhat,y_val)))

    '''Classification Case'''
    '''Needs traditional trianing first'''
    history = train(model,max_len=1000, batch_size=32,epochs=1)
    alpha=0.1
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    softmax_outputs=model.predict(val_gen)
    #print(softmax_outputs)
    #softmax_outputs = np.argmax(softmax_outputs, axis=1)
    q_yhat=calculate_qyhat_naive_classification(softmax_outputs,x_val,y_val,alpha)
    print(q_yhat)

    conf_sets=get_confsets_naive(model,x_val,y_val,q_yhat)
    calculate_class_coverage(conf_sets,y_val)


    '''Class Balanced Classification Case'''
    '''For CP WITHOUT ADV. Samples'''
    '''Needs traditional trianing first'''
    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw)#TEST: PASS!
    history = train(model,max_len=1000, batch_size=32,epochs=25)
    d_alpha_q_yhats=Calculate_q_yhats(model,x_val, y_val,alpha=0.05)
    print(d_alpha_q_yhats)
    conf_sets=ConformalPrediction_classbalanced(model,x_test,y_test,d_alpha_q_yhats,adv_attack=False)
    #Per class coverage --> needs to be called with 'y_true' only
    x_true, y_true = get_y_true()
    coverages = calculate_class_bal_coverage(conf_sets,y_true)
    #replace any nans with 0.0
    s=np.isnan(coverages)
    coverages[s]=0.0
    print(coverages)
    #Overall Coverage
    print(calculate_class_coverage(conf_sets,y_true))
    calculate_class_coverage_v1(conf_sets,y_true)

    '''For CP on ADV. Samples'''
    d_alpha_q_yhats=Calculate_q_yhats(model,x_val, y_val,alpha=0.05)
    print(d_alpha_q_yhats)
    CP_x_test = adv_y_test
    conf_sets=ConformalPrediction_classbalanced(model,CP_x_test,y_test,d_alpha_q_yhats,adv_attack=True)
    #Per class coverage --> needs to be called with 'y_true' only
    x_true, y_true = get_y_true()
    coverages = calculate_class_bal_coverage(conf_sets,y_true)
    #replace any nans with 0.0
    s=np.isnan(coverages)
    coverages[s]=0.0
    print(coverages)
    #Overall Coverage
    print(calculate_class_coverage(conf_sets,y_true))
    calculate_class_coverage_v1(conf_sets,y_true)


    #Todo: 
    # (1) adversarial retraining with adv_y_test i.e. adversarial samples
    # (2) rerun the Class blanced conformal prediction
    # (3) If (2) runs OK, then transform the multi-class outputs to binary for 'anomalous'
    '''Adversarial Retraining'''
    '''Case 1: 100% Adversarial Samples''' #--->not really 100%, the ADV PER is controlled above
    batch_size=64
    epochs=2
    verbose=1
    max_len=1000
    history = model.fit(
        adv_y_test, y_test,
        steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
        epochs=epochs, 
        verbose=verbose, 
        #validation_data=utils.data_generator(x_val, y_val, True, max_len, batch_size),
        validation_steps=int(np.ceil(len(x_val)/batch_size))
    )
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    adv_y_pred = model.predict(val_gen)
    adv_y_pred = np.argmax(adv_y_pred, axis=1)
    '''Compare results'''
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    y_val_n = [y for (x,y) in val_gen]
    y_val_n = list(itertools.chain.from_iterable(y_val_n))
    y_val_n = np.argmax(y_val_n, axis=1)
    testResults(y_val_n,adv_y_pred)

    '''Case 2: ADV_PER% Adversarial Samples'''
    adv_set_x, adv_set_y = generate_adv_set(adv_y_test, y_test, x_train, y_train,percentage=ADV_PER)#does this 'percentage' even matter
    len(adv_set_x)
    len(adv_set_y)
    batch_size=64
    epochs=500
    verbose=1
    max_len=1000
    history = model.fit(
        #adv_set_x, adv_set_y,
        utils.data_generator(adv_set_x, adv_set_y, True, max_len, batch_size,shuffle=True),
        steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
        epochs=epochs, 
        verbose=verbose, 
        #validation_data=utils.data_generator(x_val, y_val, True, max_len, batch_size),
        validation_steps=int(np.ceil(len(x_val)/batch_size))
    )
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    adv_y_pred = model.predict(val_gen)
    adv_y_pred = np.argmax(adv_y_pred, axis=1)
    '''Compare results'''
    val_gen = utils.data_generator(x_val, y_val, False, max_len=1000, batch_size=64,shuffle=True)
    y_val_n = [y for (x,y) in val_gen]
    y_val_n = list(itertools.chain.from_iterable(y_val_n))
    y_val_n = np.argmax(y_val_n, axis=1)
    testResults(y_val_n,adv_y_pred)
    len(adv_y_test)
    len(x_test)
    len(x_train)
    
    ##########################################################################################
    ############################Clustered Class-Conditional Conformal Prediction##############
    ##########################################################################################
    '''Clustered Class-Conditional Conformal Prediction'''
    import clustered_class_conformal_utils as cccp


    '''Before Adv. Attack'''
    history = train(model,max_len=1000, batch_size=32, epochs=2)
    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw)#TEST: PASS!
    
    
    test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
    y_pred = model.predict(test_gen)#,steps=len(y_test)/64,verbose=1) 
    

    


    '''After Adv. Attack'''
    from cw import CW
    #get 30% adv samples --> 30% test set size
    ADV_PER = 0.4
    VAL_PER = 0.1
    TRAIN_PER = 1.0-VAL_PER-ADV_PER
    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw,train_ratio=TRAIN_PER,test_ratio=ADV_PER)#TEST: PASS!
    x_true, y_true = get_y_true()

    atk = CW(model, x_train, y_train, x_val, y_val, x_test, y_test, c=1, kappa=0, steps=100, lr=0.01)
    atk.set_model_training_mode(model_training=True, batchnorm_training=False, dropout_training=False)
    adv_y_test = atk(x_true, y_true)

    adv_y_test = adv_y_test.detach().numpy()

    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw)#TEST: PASS!
    
    
    test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
    y_pred = model.predict(test_gen)#,steps=len(y_test)/64,verbose=1) 
    


    '''After Adv. Retraining'''
    adv_set_x, adv_set_y = generate_adv_set(adv_y_test, y_test, x_train, y_train,percentage=ADV_PER)#does this 'percentage' even matter
    len(adv_set_x)
    len(adv_set_y)
    batch_size=64
    epochs=2
    verbose=1
    max_len=1000
    history = model.fit(
        #adv_set_x, adv_set_y,
        utils.data_generator(adv_set_x, adv_set_y, True, max_len, batch_size,shuffle=True),
        steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
        epochs=epochs, 
        verbose=verbose, 
        #validation_data=utils.data_generator(x_val, y_val, True, max_len, batch_size),
        validation_steps=int(np.ceil(len(x_val)/batch_size))
    )
    x_train, x_test, y_train, y_test = utils.my_train_test_split(input_raw)#TEST: PASS!
    
    
    test_gen = utils.data_generator(x_test, y_test, False, max_len=1000, batch_size=64,shuffle=True)
    y_pred = model.predict(test_gen)#,steps=len(y_test)/64,verbose=1) 
    

    '''Get y_true value for comparison'''
    x_true, y_true = get_y_true()

    alpha = 0.05
    #softmax_Scores = (num_instances, num_classes) array i.e. y_pred before argmax()
    softmax_scores = y_pred
    scores_all = 1 - softmax_scores
    n_avg = 500# Average number of examples per class 
    cal_scores_all, cal_labels, val_scores_all, val_labels = cccp.random_split(scores_all, y_true, n_avg)
    qhats, preds, class_cov_metrics, set_size_metrics = cccp.clustered_conformal(cal_scores_all, cal_labels,
                                                                        alpha,
                                                                        val_scores_all=val_scores_all, 
                                                                        val_labels=val_labels)
    print(class_cov_metrics)