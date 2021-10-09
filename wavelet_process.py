#!/usr/bin/env python3
# wavelet_process.py

# Copyright (c) 2021 Timothy Morris Phd

import sys
from random import shuffle # Randomly order data in lists
import numpy as np
import pandas as pd
from wavelet import wavelet_transform
from wavelet import linearize
from wavelet import load_data

try:
    SCALE = int(sys.argv[1])
except:
    SCALE = 50

def data_shuffle(data,classification):
    length = len(classification)
    ind = []
    for ii in range(length):
        ind.append(ii)
    shuffle(ind)
    data_out=[]
    class_out=[]
    for ii in ind:
        data_out.append(data[ii])
        class_out.append(classification[ii])
    return (data_out,class_out)

## Read Data
trainset = "mitbih_train.csv"
testset = "mitbih_test.csv"
#verify_ab = "ptbdb_abnormal.csv"
#verify_norm = "ptbdb_normal.csv"
lc = 0
with open(trainset,mode='r') as csv_file:
    for line in csv_file:
        lc+=1
with open(testset,mode='r') as csv_file:
    for line in csv_file:
        lc+=1

print("Loading Training data")
Train, Train_class = load_data(trainset, convert_class=False)
print("Loading Testing data")
Test, Test_class = load_data(testset)
print("Shuffling .",end="")
(Train,Train_class)=data_shuffle(Train,Train_class)
print(". ",end="")
(Test,Test_class)=data_shuffle(Test,Test_class)
print(". done")

wavelets = ['mexh', 'morl', 'gaus5']

for wavelet in wavelets:
    print()
    print("Processing for wavelet:" + wavelet)
    line_count = 0
    datfile=open("Train." + wavelet + ".dat",'w')
    clfile=open("Train." + wavelet + ".cls",'w')
    for ii in range(len(Train)):
        if line_count % 100 == 0:
            datfile.flush()
            clfile.flush()
            print("\rCalculating Waveletes: "+str(line_count/lc*100000//10/100)+" % done  ",end="")
        df_data = pd.DataFrame(Train[ii])
        N= df_data.shape[0]
        t0 = 0
        dt = 1
        time = np.arange(0,N)*dt+t0
        signal = df_data.values.squeeze()
        scales = np.arange(1,33,2) #[1,3,...,31]
        transformed_wavelet, frequencies = wavelet_transform(time,signal,scales,wavelet,SCALE)
        linearized_wavelet = linearize(transformed_wavelet)
        outline=""
        for c in linearized_wavelet:
            outline+=str(c)+","
        datfile.write(outline[:-1]+"\n")
        clfile.write(str(Train_class[ii])+"\n")
        line_count+=1
    datfile.flush()
    clfile.flush()
    datfile.close()
    clfile.close()
    datfile=open("Test." + wavelet + ".dat",'w')
    clfile=open("Test." + wavelet + ".cls",'w')
    for ii in range(len(Test)):
        if line_count % 100 == 0:
            datfile.flush()
            clfile.flush()
            print("\rCalculating Waveletes: "+str(line_count/lc*100000//10/100)+" % done  ",end="")
        df_data = pd.DataFrame(Train[ii])
        N= df_data.shape[0]
        t0 = 0
        dt = 1
        time = np.arange(0,N)*dt+t0
        signal = df_data.values.squeeze()
        scales = np.arange(1,33,2) #[1,3,...,31]
        transformed_wavelet, frequencies = wavelet_transform(time,signal,scales,wavelet,SCALE)
        linearized_wavelet = linearize(transformed_wavelet)
        outline=""
        for c in linearized_wavelet:
            outline+=str(c)+","
        datfile.write(outline[:-1]+"\n")
        clfile.write(str(Train_class[ii])+"\n")
        line_count+=1
    datfile.flush()
    clfile.flush()
    datfile.close()
    clfile.close()
