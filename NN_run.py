#!/usr/bin/env python3
# NN.py

import numpy as np # Linear Algebra
import pywt # wavelets
import pandas as pd # Data tools
import csv # CSV reading info
from random import shuffle # Randomly order data in lists
from sklearn.neural_network import MLPClassifier
import time
import pickle

X_test = []
Y_test = []
lc=0
with open("Test.morl.cls",mode='r') as classfile:
    for line in classfile:
        Y_test.append(int(float(line)))
        lc+=1
row_no=0
with open("Test.morl.dat",mode='r') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=",")
    for row in csv_reader:
        if row_no % 500 == 0:
            print("\r"+str(row_no/lc*100)+"                           ",end="")
        cur_data =[] # data from this row
        for d_item in range(0,len(row)):
            cur_data.append(float(row[d_item]))
        X_test.append(cur_data)
        row_no+=1
print()

clf = pickle.load(open("NN.027.model",'rb'))

output_counts={  # Actual:predicted
    0:{0:0,1:0,2:0,3:0,4:0},
    1:{0:0,1:0,2:0,3:0,4:0},
    2:{0:0,1:0,2:0,3:0,4:0},
    3:{0:0,1:0,2:0,3:0,4:0},
    4:{0:0,1:0,2:0,3:0,4:0}
}

for ii in range(len(Y_test)):
    if ii % 500 == 0:
        print("\r"+str(ii/lc*100)+"                           ",end="")
    pred=clf.predict([X_test[ii]])
    act =Y_test[ii]
    output_counts[act][pred[0]]+=1

print()
lines=[]
out=open("characterization.txt",'w')
for act in output_counts.keys():
    cur_line=""
    for pred in output_counts[act].keys():
        cur_line += str(output_counts[act][pred])+","
        print(str(output_counts[act][pred])+"\t",end="")
    cur_line+="\n"
    print()
    lines.append(cur_line[:-1])

for line in lines:
    out.write(line)


