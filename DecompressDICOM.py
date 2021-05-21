import argparse
import os
import sys

# decompress dicom files (SGS surgery compressed DICOM)
exepath=r'dcmdjpeg.exe' # make sure dcmtk was installed (C:\dcmtk-3.6.5-win64-dynamic\bin\)

def get_all_file(rawdir):
    allfile = []
    allfilelist=os.listdir(rawdir)
    for f in allfilelist:
        filepath=os.path.join(rawdir,f)
        if os.path.isdir(filepath):
                get_all_file(f)
        allfile.append(filepath)
    return allfile


# path of compressed files and output path
data_path = "./SurgeryData/"
outpath = "./Decompressed/"


import os
subfolders = [f.path for f in os.scandir(data_path) if f.is_dir() ]  


# decompress each of the DICOM file
for folder in subfolders:
    allfile = get_all_file(folder)
    outfolder = outpath + folder[len(data_path):] + '\\'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for file in allfile:
        filepath, filename = os.path.split(file)
        inputpath=os.path.join(folder,filename)
        outputpath=os.path.join(outfolder,filename)
        os.system(exepath+' '+inputpath + ' '+outputpath)
    print('Processed '+ folder[len(data_path):])


