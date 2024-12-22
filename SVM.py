# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:27:47 2024

@author: gary
"""
from tensorflow import keras
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import scipy.io as sio 
from sklearn.datasets import make_blobs
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
def calRecall(dec,gt):
    recall = np.zeros(dec.shape[1])
    N_AP = np.zeros(dec.shape[1]) # actual positive
    N_TP = np.zeros(dec.shape[1]) # true positive
    for j in range(dec.shape[1]):
        N_AP[j] = np.count_nonzero(gt[:,j])
        N_TP[j] = np.count_nonzero(np.multiply(gt[:,j],dec[:,j]))
        recall[j] = N_TP[j] / N_AP[j]
    return recall
def ExactMatchRate(dec,gt):
    emr = 0
    for i in range(dec.shape[0]):
        if np.all(gt[i] == dec[i]):
            emr += 1
    return emr/dec.shape[0]
def ThresholdDecision(pred):
    #pred = np.reshape(pred, (pred.shape[0],pred.shape[1]))
    dec = np.zeros((pred.shape[0],pred.shape[1]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j] > 0.5:
                dec[i,j] = 1
    return dec
def calFalseAlarmRate(dec,gt):
    far = np.zeros(dec.shape[1])
    N_AN = np.zeros(dec.shape[1]) # actual negative
    N_FP = np.zeros(dec.shape[1]) # false positive
    for j in range(dec.shape[1]):
        N_AN[j] = gt.shape[0] - np.count_nonzero(gt[:,j])
        N_FP[j] = np.count_nonzero((gt[:,j]==0) & (dec[:,j]==1))
        far[j] = N_FP[j] / N_AN[j]
    return far
def calPrecision(dec,gt):
    precision = np.zeros(dec.shape[1])
    N_PP = np.zeros(dec.shape[1]) # predict positive
    N_TP = np.zeros(dec.shape[1]) # true positive
    for j in range(dec.shape[1]):
        N_PP[j] = np.count_nonzero(dec[:,j])
        N_TP[j] = np.count_nonzero(np.multiply(gt[:,j],dec[:,j]))
        precision[j] = N_TP[j] / N_PP[j]
    return precision

def calF1(dec,gt):
    F1 = np.zeros(dec.shape[1])
    recall = calRecall(dec,gt)
    prec = calPrecision(dec,gt)
    F1 = 2 / (np.power(recall,-1) + np.power(prec,-1))
    return F1
n = pd.read_csv("n.csv")
a1 = pd.read_csv("eb.csv")
b1 = pd.read_csv("eu1.csv")
c1 = pd.read_csv("eu2.csv")
mamb1 = pd.read_csv("mBm1.csv")
mamc1 = pd.read_csv("mBm2.csv")
mbmc1 = pd.read_csv("m1m2.csv")
mambmc1 = pd.read_csv("mBm1m2.csv")
a = a1[0:300]
b = b1[0:300]
c = c1[0:300]
mamb = mamb1[0:60]
mamc = mamc1[0:60]
mbmc = mbmc1[0:60]
mambmc = mambmc1[0:60]
train = np.concatenate([n,a,b,c,mamb,mamc,mbmc,mambmc])

label_train = pd.read_csv("label_train.csv")

train_labela = keras.utils.to_categorical(label_train, 3)
train_labelz = train_labela[:,:,0]

tn = pd.read_csv("tn.csv")
a = a1[300:400]
b = b1[300:400]
c = c1[300:400]
mamb = mamb1[60:100]
mamc = mamc1[60:100]
mbmc = mbmc1[60:100]
mambmc = mambmc1[60:100]
test = np.concatenate([tn,a,b,c,mamb,mamc,mbmc,mambmc])
label_test = pd.read_csv("label_test.csv")

label_teata = keras.utils.to_categorical(label_test, 3)
label_teatz = label_teata[:,:,0]

# Create the SVM
svm = LinearSVC(random_state=42)

# Make it an Multilabel classifier
multilabel_classifier = MultiOutputClassifier(svm, n_jobs=-1)

# Fit the data to the Multilabel classifier
multilabel_classifier = multilabel_classifier.fit(train, train_labelz )

# Get predictions for test data
preds = multilabel_classifier.predict(test)


emr = ExactMatchRate(preds, label_teatz) # train_label, x_pred
recall = calRecall(preds, label_teatz)
far = calFalseAlarmRate(preds, label_teatz)
precision = calPrecision(preds, label_teatz)
F1 = calF1(preds, label_teatz)
print('emr',emr)
print('recall',recall)
print('far',far)
print('precision',precision)
print('F1',F1)