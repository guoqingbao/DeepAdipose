import numpy as np
import pandas as pd
import SimpleITK as itk # simpleitk is required


import gc
gc.enable()

# path of dicom
data_path = "./data/dicom/" # folder contains all dicoms
output_path = "./data/tiff/" # output folder for tiff files

data_path = './DICOMs/'
output_path = './OutputTiff/'


import os
backIndex = 0
path_lst = [f.path +"/" for f in os.scandir(data_path) if f.is_dir() ]  

#convert dicom to tiffs
import yaml
d = {}
index = 0
cur =0

total = len(path_lst)
for item in path_lst:
    print(item)
    if index + 1 < backIndex:
        index = index + 1
        continue
        
    print( "Reading Dicom directory:", item )
    lef = item[len(data_path):]
    pa = lef[0:lef.find("/")]
    if os.path.exists(output_path + pa + ".tif"):
        print("Skipping ", item)
    else:
        reader = itk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(item)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()

        
        for k in reader.GetMetaDataKeys(slice=0):
            v = reader.GetMetaData(key=k,slice=0)
            d[k] = v

        np.save(output_path + pa + ".npy", np.array(d))

        size = image.GetSize()

        print( "Image size:", size[0], size[1], size[2] )


        a = yaml.load(str(np.array(d)))
        images = itk.GetArrayFromImage(image).astype(np.int16)
        images[images < -1024] = -1024
        if a['0020|0013'] == '1 ': #do not need to flip
            del image
            image = itk.GetImageFromArray(images)
            print( "Writing image:", output_path + pa + ".tif" )
            itk.WriteImage( image, output_path + pa + ".tif" )
            del reader
            del image
            del images
        else:
            # some of the cases need to flip
            print( "Flipping image...")
            images2 = itk.GetImageFromArray(np.flip(images,0))
            print( "Writing image:", output_path + pa + ".tif" )

            itk.WriteImage( images2, output_path + pa + ".tif" )
            del reader
            del image
            del images
            del images2
        gc.collect()
        print("Writing down.")
    index += 1
    if cur != int(index/total *100):
        cur = int(index/total *100)
        print("Progress {} %, curIndex {}".format(cur, index))
    


# import pickle
# for item in path_lst:
# #     print(item)
#     lef = item[len(data_path):]
#     pa = lef[0:lef.find("/")]
#     print(output_path + pa + ".npy")
#     pkl = np.load(open(output_path + pa + ".npy", "rb"), allow_pickle=True)

#     pickle.dump(pkl, open(output_path + pa + ".npy2","wb"), protocol=2)



