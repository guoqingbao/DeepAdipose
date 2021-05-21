from __future__ import print_function
import os, time
import collections
import platform
import csv
import SimpleITK as sitk
import six

from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, featureextractor
import numpy as np 
# import matplotlib.pyplot as plt 
import imp
import datetime
import pandas as pd
import gc
gc.enable()

# !pip install radiomics

import radiomics

# data_path = "./data/"
# tif_path = mask_path =  './data/tiff_sgs/'
# output_path = './data/block_sgs/'
# frame = pd.read_csv(data_path + 'sgs_segment_results.csv') # segmentation results of SGS surgery

data_path = './OutputTiff'
tif_path = mask_path = data_path
output_path = data_path 
frame = pd.read_csv(data_path + 'results.csv')

ids = [item for item in list(frame.ID)]

# same features as in Volkswagen cohort
innerHeader = pd.read_excel(data_path + "allHeaders.xlsx")
header = innerHeader.columns.tolist()[:-9]

# extract texture features from each SGS CT studies (include preoperative and postoperative studies)
FEATURE_TYPES = {1:"shape", 2:"firstorder", 3:"glcm", 4:"glrlm", 5:"glszm", 6:"wavelet-glcm", 7:"wavelet-glrlm"}
import random

IS_OUTER_FAT = False

TASK_LIST = [2,3,4,5,6,7]

POSITION = 5
first = True
FINISHED_ID = ""

try:
    with open(output_path + "blocks.txt", 'r') as sf:
        FINISHED_ID = sf.readline()
        print(FINISHED_ID)
        if FINISHED_ID != "":
            first = False;
except:
    pass

Found = False
positions = []

if IS_OUTER_FAT:
    CSV_FILE = "block_outer.csv"
else:
    CSV_FILE = "block_inner.csv"
    
for index, row in frame.iterrows():
    
    if FINISHED_ID!="" and Found==False and str(row["ID"]) != FINISHED_ID:
        continue
    elif str(row["ID"]) == FINISHED_ID:
        Found = True
        print("Continue from {}".format(row["ID"]))
        continue
        
    file = output_path+str(row["ID"])+".tif"
    if os.path.exists(file): 
        mask_file = mask_path+str(row["ID"])+"_mask.nrrd"
        if not os.path.exists(mask_file):
            print("mask file not found: "+ mask_file) 
            continue
    else:
        print("file not found: "+ file)
        continue
        
    image_block = sitk.ReadImage(file)    
    mask = sitk.ReadImage(mask_file)
    mask_s = sitk.GetArrayFromImage(mask)
    if IS_OUTER_FAT:
        mask_s[mask_s!=2] = 0
        mask_s[mask_s==2] = 1
    else:
        mask_s[mask_s!=5] = 0
        mask_s[mask_s==5] = 1
        
#     pos, endpos = row["Diaphragm Index"],row["PubicSymphysis Index"] 
    mask_block = sitk.GetImageFromArray(mask_s)
    del mask
    del mask_s
    
    print("\r\n*******************************************************\n")
    print("Case {}".format(row["ID"]))

    settings = {}
    spacing = ['1.0','1.0','1.0']  
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
     
    values = {'ID':row["ID"]}
    for FEATURE_ID in TASK_LIST:
#         print("Start analyze feature: {}...".format(FEATURE_TYPES[FEATURE_ID]))
        if FEATURE_ID == 6:
            paramPath = output_path + 'Params-glcm.yaml'
        else:
            paramPath = output_path + 'Params-glrlm.yaml'

        selected_feature = None
        if FEATURE_TYPES[FEATURE_ID] == "firstorder":
            extractor = firstorder.RadiomicsFirstOrder(image_block, mask_block, **settings)
        elif FEATURE_TYPES[FEATURE_ID] == "glcm":
            extractor = glcm.RadiomicsGLCM(image_block, mask_block, **settings)
        elif FEATURE_TYPES[FEATURE_ID] == "glrlm":
            extractor = glrlm.RadiomicsGLRLM(image_block, mask_block, **settings)
        elif FEATURE_TYPES[FEATURE_ID] == "glszm":
            extractor = glszm.RadiomicsGLSZM(image_block, mask_block, **settings)
        elif FEATURE_TYPES[FEATURE_ID] == "wavelet-glcm" or FEATURE_TYPES[FEATURE_ID] == "wavelet-glrlm":
            extractor = featureextractor.RadiomicsFeatureExtractor(paramPath) 
        else:
            raise Exception("Invalid feature selected!")
            

#         print('Calculating feature {} '.format(FEATURE_TYPES[FEATURE_ID]))

        featureValues= {}
        if FEATURE_ID == 6 or FEATURE_ID == 7:
            featureValues = extractor.execute(image_block,mask_block)   
        elif extractor != None:
            extractor.enableAllFeatures()
            extractor.execute()
            featureValues = extractor.featureValues
        else:
             raise Exception("Invalid feature configuration!")
                
        new_values = {k.replace('wavelet-',''):v for k, v in featureValues.items() if not k.startswith('general_')} 

        new_values = {k:v for k, v in new_values.items() if k.find('diagnostics_')<0} 
        
        
        nm = FEATURE_TYPES[FEATURE_ID].upper()
        if FEATURE_ID == 6 or FEATURE_ID == 7:
            nm = 'WAVELET'
        new_values = { nm+ '_' +k:v for k, v in new_values.items()} 
        values.update(new_values)

    del values['GLCM_MCC']
#         print('Calculated feature {} '.format(FEATURE_TYPES[FEATURE_ID]))
#         for (key, val) in six.iteritems(featureValues):
#             print('  ', key, ':', val)

    with open(output_path + CSV_FILE, 'a', newline='') as f:
#         print('Write')
        writer = csv.DictWriter(f, fieldnames=header)
        if first:
            writer.writeheader()
        writer.writerow(values)
        f.flush()
        first = False
            
    del image_block
    del mask_block
    del extractor
    gc.collect()

    with open(output_path + "blocks.txt", 'w+') as f:
        f.write(str(row["ID"]))
        f.flush()


