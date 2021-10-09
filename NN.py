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

X_train = []
Y_train = []
lc=0
with open("Train.cls",mode='r') as classfile:
    for line in classfile:
        lc+=1
        Y_train.append(int(float(line)))
        if lc == 20000:
            break
row_no=0
with open("Train.dat",mode='r') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=",")
    for row in csv_reader:
        if row_no % 500 == 0:
            print("\r"+str(row_no/lc*100)+"                           ",end="")
        cur_data =[] # data from this row
        for d_item in range(0,len(row)):
            cur_data.append(float(row[d_item]))
        X_train.append(cur_data)
        row_no+=1
        if row_no ==20000:
            break
print()
X_test = []
Y_test = []
lc=0
with open("Test.cls",mode='r') as classfile:
    for line in classfile:
        Y_test.append(int(float(line)))
        lc+=1
row_no=0
with open("Test.dat",mode='r') as csv_file:
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

NN_params={
    #'hidden_layer_size':[(775),(775,31)],
    'hidden_layer_size':[(100),(16,16),(100,16)],
    'activation':['relu','tanh'],
    'alpha':[0.00001,0.0001,0.001,0.01,0.1,1]
}

overview_file=open("model_desc.txt",'w')
model_no=1000

for h_l in NN_params['hidden_layer_size']:
    for act in NN_params['activation']:
        for a in NN_params['alpha']:
            print("HL:"+str(h_l)+" activation:"+str(act)+" alpha:"+str(a))
            clf = MLPClassifier(hidden_layer_sizes=h_l,activation=act,alpha=a)
            t_start=time.perf_counter()
            clf.fit(X_train,Y_train)
            t_end=time.perf_counter()
            train_score = clf.score(X_train,Y_train)
            test_score = clf.score(X_test,Y_test)
            train_time = t_end - t_start
            model_file = "NN."+str(model_no)[1:]+".model"
            pickle.dump(clf,open(model_file,'wb'))
            overview_file.write(str(model_file)+" ({}, {}, {}) - train, test score: \t {:.5f} - {:.5f} \t in {:.2f} s\n".format(h_l,act,a,train_score,test_score,train_time))
            print("For ({}, {}, {}) - train, test score: \t {:.5f} - {:.5f} \t in {:.2f} s".format(h_l,act,a,train_score,test_score,train_time))
            model_no+=1



