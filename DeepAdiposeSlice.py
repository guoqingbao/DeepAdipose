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


output_path = './Results/'
model_path = output_path + "Models/"
data_path = './Data/'


# load training data
allSlicedata = pd.read_excel(data_path + 'allSliceData.xlsx',index_col=0)
allClinical = pd.read_excel(data_path + "allClinical.xlsx",index_col=0)
allSegment = pd.read_excel(data_path + "allSegment.xlsx",index_col=0)


block_data = pd.read_excel(data_path + "allBlockData.xlsx",index_col=0)
data = block_data.sample(frac=1,random_state=11)
study_frame,test_frame, c,d = train_test_split(data.ix[:,:-1],data.ix[:,-1],stratify=data.ix[:,'IsMS'], test_size=0.2, random_state=9)
slice_study = allSlicedata[allSlicedata.ID.isin(study_frame.ID.tolist())]
slice_study.ix[:,1:-8] = (slice_study.ix[:,1:-8] - slice_study.ix[:,1:-8].mean())/slice_study.ix[:,1:-8].std()
slice_study.reset_index(drop=True,inplace=True)


# top ten radiomics features + gender
op = ['GLCM_Imc2', 'GLRLM_RunLengthNonUniformity', 'GLSZM_GrayLevelNonUniformity',
       'GLSZM_GrayLevelVariance', 'WAVELET_LLL_glcm_Idm', 'WAVELET_LLL_glcm_Id','WAVELET_LLH_glcm_Idm', 'WAVELET_LLH_glcm_Id',
       'WAVELET_LLH_glrlm_RunLengthNonUniformity', 'WAVELET_HHH_glrlm_RunEntropy', 'Gender']


# the neural network model (same as used in block)
from keras.layers import GaussianNoise, GaussianDropout
from keras.optimizers import RMSprop,Adam
from keras.regularizers import l2
def NeuralModel(input_dim):
    model = Sequential()
    model.add(Dense(1024, input_dim= input_dim, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.05), name="DenseLayer_1024"))
    model.add(GaussianNoise(0.02,name="GaussianNoise_0.02"))

    model.add(Dense(512,  activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(0.03), name="DenseLayer_512"))
    model.add(GaussianDropout(0.05, name="GaussianDropout_0.05"))   

    model.add(Dense(256,  activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(0.03), name="DenseLayer_256"))
    
    model.add(Dense(128,  activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(0.03), name="DenseLayer_128"))

    model.add(Dense(64,  activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(0.03), name="DenseLayer_64"))
    model.add(GaussianNoise(0.01,name="GaussianNoise_0.01"))

    model.add(Dense(2,activation='softmax', kernel_initializer='he_normal',kernel_regularizer=l2(0.03), name='Output'))

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model


model_json = NeuralModel(len(op)).to_json()
with open(model_path +"model_slice.json", "w") as json_file:
     json_file.write(model_json)


slice_study_shuffle = slice_study.sample(frac=1,random_state=9)
slice_study_shuffle.reset_index(drop=True, inplace=True)


# train and save the model under cross-validation

from keras.layers import GaussianNoise, GaussianDropout
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

ftypes = ['IsMS','IsCO','IsVO','IsCTVO','IsIR']

for ftype in ftypes:  
    index = 0
    mcp_save = ModelCheckpoint(model_path + ftype[2:] + '_slice_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    cv = StratifiedKFold(n_splits=10,  shuffle=True, random_state=9)
    print(ftype)
    for tr, te in cv.split(slice_study_shuffle.ix[:,op], slice_study_shuffle.ix[:,'IsMS']):  
        x_train, x_test, y_train, y_test = slice_study_shuffle.ix[tr,op], slice_study_shuffle.ix[te,op],slice_study_shuffle.ix[:,ftype][tr], slice_study_shuffle.ix[:,ftype][te]
        x_train.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        model = NeuralModel(len(op))
        history = model.fit(x_train, to_categorical(y_train), validation_data=(x_test, to_categorical(y_test)), epochs=200, batch_size=128, verbose=0, callbacks=[mcp_save])  
        pred = model.predict(x_test)
        print(accuracy_score(np.argmax(pred,axis=1), y_test))
        index += 1


