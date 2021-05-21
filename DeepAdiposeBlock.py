import os, time
import collections
import platform
import numpy as np 
import matplotlib.pyplot as plt 
import keras
import warnings;
warnings.filterwarnings('ignore');
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.pyplot import cm
from keras.models import Model
from keras.models import Sequential, load_model
from keras.optimizers import SGD 
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
# init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# load data
output_path = './Results/'
model_path = output_path + "Models/"
data_path = './Data/'
allClinical = pd.read_excel(data_path + "allClinical.xlsx",index_col=0)
segment = pd.read_excel(data_path + "allSegment.xlsx",index_col=0)
block_data = pd.read_excel(data_path + "allBlockData.xlsx",index_col=0)


# load training data
data = block_data.sample(frac=1,random_state=11)
study_frame,test_frame, c,d = train_test_split(data.ix[:,:-1],data.ix[:,-1],stratify=data.ix[:,'IsMS'], test_size=0.2, random_state=9)
block_study = data[data.ID.isin(study_frame.ID.tolist())]

block_study.ix[:,1:-8] = (block_study.ix[:,1:-8] - block_study.ix[:,1:-8].mean())/block_study.ix[:,1:-8].std()
block_study.reset_index(drop=True, inplace=True)


# top ten radiomics features + gender
op = ['GLCM_Imc2', 'GLRLM_RunLengthNonUniformity', 'GLSZM_GrayLevelNonUniformity',
       'GLSZM_GrayLevelVariance', 'WAVELET_LLL_glcm_Idm', 'WAVELET_LLL_glcm_Id','WAVELET_LLH_glcm_Idm', 'WAVELET_LLH_glcm_Id',
       'WAVELET_LLH_glrlm_RunLengthNonUniformity', 'WAVELET_HHH_glrlm_RunEntropy',  'Gender']


# the neural network model
from keras.layers import GaussianNoise, GaussianDropout
from keras.optimizers import RMSprop,Adam
from keras.regularizers import l1,l2

def NeuralModel(input_dim):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.05),name="DenseLayer_1024"))
    model.add(GaussianNoise(0.02,name="GaussianNoise_0.02"))

    model.add(Dense(512,  activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.03),name="DenseLayer_512"))
    model.add(GaussianDropout(0.05, name="GaussianDropout_0.05"))   

    model.add(Dense(256,  activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.03),name="DenseLayer_256"))
    
    model.add(Dense(128,  activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.03),name="DenseLayer_128"))

    model.add(Dense(64,  activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.03),name="DenseLayer_64"))
    model.add(GaussianNoise(0.01,name="GaussianNoise_0.01"))

    model.add(Dense(2,activation='softmax',kernel_initializer='he_normal',kernel_regularizer=l2(0.03),name='Output'))

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

# save the structure of the neural network
model_json = NeuralModel(len(op)).to_json()
with open(model_path +"model_block.json", "w") as json_file:
     json_file.write(model_json)


# train and save the model under cross-validation
from keras.layers import GaussianNoise, GaussianDropout
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
ftypes = ['IsMS','IsCO','IsVO','IsCTVO','IsIR']
  
for ftype in ftypes:  
    print(ftype)
    mcp_save = ModelCheckpoint(model_path + ftype[2:]+ '_block_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    cv = StratifiedKFold(n_splits=10,  shuffle=True, random_state=9)
  
    index = 0
    for tr, te in cv.split(block_study.ix[:,:], block_study.ix[:,-1]):  
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, verbose=0, epsilon=1e-4, mode='min')

        x_train, x_test, y_train, y_test = block_study.ix[tr,op], block_study.ix[te,op],block_study.ix[:,ftype][tr], block_study.ix[:,ftype][te]
        x_train.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        model = NeuralModel(len(op))

        history = model.fit(x_train, to_categorical(y_train),  validation_data=(x_test, to_categorical(y_test)),epochs=50, batch_size=128, verbose=0, callbacks=[mcp_save])  
        pred = model.predict(x_test)
        print(accuracy_score(np.argmax(pred,axis=1), y_test))
        index += 1




