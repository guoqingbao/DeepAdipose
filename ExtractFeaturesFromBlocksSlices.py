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

data_path = "./data/" 
frame = pd.read_csv(data_path + 'segment_results.csv')

tif_path = mask_path = data_path + 'tiff/'
output_path = "./data/block/"

ids = [int(item[item.find('_')+1:]) for item in list(frame.ID)]

frame.ix[:,'BKID'] = frame.ID
frame.ix[:,'ID'] = ids

def getpos(Diaphragm,PubicSymphysis):
    pos = int((PubicSymphysis-Diaphragm) * POSITION/10) + int(Diaphragm) 
    endpos = int(Diaphragm) + pos + int((PubicSymphysis-Diaphragm)/10) 
    if endpos + 1 >=PubicSymphysis:
        endpos = PubicSymphysis -1
    return pos, endpos

# Extract blocks from CTs (umbilicus block)
POSITION = 5
for index, row in frame.iterrows():
    file = tif_path+str(row["BKID"])+".tif"
    if os.path.exists(file): 
        if os.path.exists(output_path+str(row["ID"])+".tif"):
            print("skip "+ file)
            continue
    else:
        print("file not found: "+ file)
        continue
        
    image = sitk.ReadImage(file)    
    image_s = sitk.GetArrayFromImage(image)

    pos, endpos = getpos(row["Diaphragm Index"],row["PubicSymphysis Index"])
        
    img = sitk.GetImageFromArray(image_s[pos:endpos])

    sitk.WriteImage(img, output_path + str(row["ID"])+'.tif') 

    del img
    del image_s
    gc.collect()
    print("Extracted block from " + str(row["ID"]))


innerHeader = pd.read_excel(data_path + "full_block_features.xlsx")
header = innerHeader.columns.tolist()
header.append("StartIndex")


# Extract features from umbilicus CT block 

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
        mask_file = mask_path+str(row["BKID"])+"_mask.nrrd"
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
        
    pos, endpos = getpos(row["Diaphragm Index"],row["PubicSymphysis Index"]) 
    mask_block = sitk.GetImageFromArray(mask_s[pos:endpos])
    del mask
    del mask_s
    
    print("\r\n*******************************************************\n")
    print("Case {}, pos {} - {}".format(row["ID"], pos, endpos))

    settings = {}
    spacing = ['1.0','1.0','1.0']  
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
     
    values = {'ID':row["ID"], "StartIndex":pos}
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
            extractor = featureextractor.RadiomicsFeaturesExtractor(paramPath)
        else:
            raise Exception("Invalid feature selected!")

#         print('Calculating feature {} '.format(FEATURE_TYPES[FEATURE_ID]))

        featureValues= {}
        if FEATURE_ID == 6 or FEATURE_ID == 7:
            featureValues = extractor.execute(image_block,mask_block)   
        elif extractor != None:
            extractor.enableAllFeatures()
            extractor.calculateFeatures()
            featureValues = extractor.featureValues
        else:
             raise Exception("Invalid feature configuration!")
        new_values = {k.replace('wavelet-',''):v for k, v in featureValues.items() if not k.startswith('general_')} 

        nm = FEATURE_TYPES[FEATURE_ID].upper()
        if FEATURE_ID == 6 or FEATURE_ID == 7:
            nm = 'WAVELET'
        new_values = { nm+ '_' +k:v for k, v in new_values.items()} 
        values.update(new_values)


#         print('Calculated feature {} '.format(FEATURE_TYPES[FEATURE_ID]))
#         for (key, val) in six.iteritems(featureValues):
#             print('  ', key, ':', val)

    with open(output_path + CSV_FILE, 'a', newline='') as f:
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

# Extract any three consecutive slices from umbilicus CT block (10 times of random position)

first = True
FINISHED_ID = ""

try:
    with open(output_path + "slices.txt", 'r') as sf:
        FINISHED_ID = sf.readline()
        print(FINISHED_ID)
        if FINISHED_ID != "":
            first = False;
except:
    pass

Found = False
positions = []

if IS_OUTER_FAT:
    CSV_FILE = "block_slices.csv"
else:
    CSV_FILE = "block_slices.csv"
    
for index, row in frame.iterrows():
    
    if FINISHED_ID!="" and Found==False and str(row["ID"]) != FINISHED_ID:
        continue
    elif str(row["ID"]) == FINISHED_ID:
        Found = True
        print("Continue from {}".format(row["ID"]))
        continue
        
    file = output_path+str(row["ID"])+".tif"
    if os.path.exists(file): 
        mask_file = mask_path+str(row["BKID"])+"_mask.nrrd"
        if not os.path.exists(mask_file):
            print("mask file not found: "+ mask_file) 
            continue
    else:
        print("file not found: "+ file)
        continue
        
    image = sitk.ReadImage(file)   
    image_s = sitk.GetArrayFromImage(image)

    mask = sitk.ReadImage(mask_file)
    mask_s = sitk.GetArrayFromImage(mask)
    if IS_OUTER_FAT:
        mask_s[mask_s!=2] = 0
        mask_s[mask_s==2] = 1
    else:
        mask_s[mask_s!=5] = 0
        mask_s[mask_s==5] = 1
        
    pos, endpos = getpos(row["Diaphragm Index"],row["PubicSymphysis Index"]) 
    print("\r\n*******************************************************\n")

    for i in range(10):
        start = random.randint(pos,endpos-5) 
        
        image_slices = sitk.GetImageFromArray(image_s[start-pos:start-pos+3])
        mask_slices = sitk.GetImageFromArray(mask_s[start:start+3])
        print("Case {}, slice pos {} - {}".format(row["ID"], start, start+3))

        settings = {}
        spacing = ['1.0','1.0','1.0']  
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  
        settings['interpolator'] = 'sitkBSpline'
        settings['verbose'] = True

        values = {'ID':row["ID"], "StartIndex":start}
        for FEATURE_ID in TASK_LIST:
    #         print("Start analyze feature: {}...".format(FEATURE_TYPES[FEATURE_ID]))
            if FEATURE_ID == 6:
                paramPath = output_path + 'Params-glcm.yaml'
            else:
                paramPath = output_path + 'Params-glrlm.yaml'

            selected_feature = None
            if FEATURE_TYPES[FEATURE_ID] == "firstorder":
                extractor = firstorder.RadiomicsFirstOrder(image_slices, mask_slices, **settings)
            elif FEATURE_TYPES[FEATURE_ID] == "glcm":
                extractor = glcm.RadiomicsGLCM(image_slices, mask_slices, **settings)
            elif FEATURE_TYPES[FEATURE_ID] == "glrlm":
                extractor = glrlm.RadiomicsGLRLM(image_slices, mask_slices, **settings)
            elif FEATURE_TYPES[FEATURE_ID] == "glszm":
                extractor = glszm.RadiomicsGLSZM(image_slices, mask_slices, **settings)
            elif FEATURE_TYPES[FEATURE_ID] == "wavelet-glcm" or FEATURE_TYPES[FEATURE_ID] == "wavelet-glrlm":
                extractor = featureextractor.RadiomicsFeaturesExtractor(paramPath)
            else:
                raise Exception("Invalid feature selected!")

    #         print('Calculating feature {} '.format(FEATURE_TYPES[FEATURE_ID]))

            featureValues= {}
            if FEATURE_ID == 6 or FEATURE_ID == 7:
                featureValues = extractor.execute(image_slices,mask_slices)   
            elif extractor != None:
                extractor.enableAllFeatures()
                extractor.calculateFeatures()
                featureValues = extractor.featureValues
            else:
                 raise Exception("Invalid feature configuration!")
            new_values = {k.replace('wavelet-',''):v for k, v in featureValues.items() if not k.startswith('general_')} 


            nm = FEATURE_TYPES[FEATURE_ID].upper()
            if FEATURE_ID == 6 or FEATURE_ID == 7:
                nm = 'WAVELET'
            new_values = { nm+ '_' +k:v for k, v in new_values.items()} 
            values.update(new_values)


    #         print('Calculated feature {} '.format(FEATURE_TYPES[FEATURE_ID]))
    #         for (key, val) in six.iteritems(featureValues):
    #             print('  ', key, ':', val)

        with open(output_path + CSV_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if first:
                writer.writeheader()
            writer.writerow(values)
            f.flush()
            first = False
            
        del image_slices
        del mask_slices
        del extractor
        
    del mask
    del mask_s
    del image
    del image_s
    gc.collect()
    
    with open(output_path + "slices.txt", 'w+') as f:
        f.write(str(row["ID"]))
        f.flush()




