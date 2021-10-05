
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import sys
from keras.datasets import mnist
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
import os
#import numpy as np
from PIL import Image
#import pickle

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve




protein_embeddings_train = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[:5222]
length_train = len(protein_embeddings_train["mean_ci_2"])
#length_train = 5222
protein_embeddings_train = [protein_embeddings_train.loc[i].values for i in range(length_train)]
#print(protein_embeddings[0])

drug_embeddings_train = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[:5222]
drug_embeddings_train = [drug_embeddings_train.loc[i].values for i in range(length_train)]
#print(drug_embeddings[0])

all_embeddings_train = np.hstack((protein_embeddings_train, drug_embeddings_train))
print(type(all_embeddings_train))

all_embeddings_train = [all_embeddings_train[i].reshape(64,1) for i in range(length_train)]
 
labels_train = pd.read_csv("celegans-data-pre.csv")["label"][:5222]
print(len(labels_train))




protein_embeddings_test = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[5222:]
protein_embeddings_test = protein_embeddings_test.reset_index(drop=True)
length_test = len(protein_embeddings_test["mean_ci_2"])
protein_embeddings_test = [protein_embeddings_test.loc[i].values for i in range(length_test)]
#print(protein_embeddings[0])

drug_embeddings_test = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[5222:]
drug_embeddings_test = drug_embeddings_test.reset_index(drop=True)
drug_embeddings_test = [drug_embeddings_test.loc[i].values for i in range(length_test)]
#print(drug_embeddings[0])

all_embeddings_test = np.hstack((protein_embeddings_test, drug_embeddings_test))
print(type(all_embeddings_test))

all_embeddings_test = [all_embeddings_test[i].reshape(64,1) for i in range(length_test)]
 
labels_test = list(pd.read_csv("celegans-data-pre.csv")["label"][5222:])










 
X_train = all_embeddings_train
y_train = np.array(labels_train)
X_test = all_embeddings_test
y_test =labels_test
print(np.array(X_train).shape)

X_train = np.array(X_train)[:, np.newaxis, :, :]
X_test = np.array(X_test)[:, np.newaxis, :, :]

print(X_test.shape)



    # load
with open("embedding-celegans-32dim-XGB.pkl", "rb") as f:
    gc = pickle.load(f)
    
y_pred_train = gc.predict(X_train) 
y_pred_train_proba = gc.predict_proba(X_train)
y_pred_train = pd.DataFrame(y_pred_train)
y_pred_train_proba = pd.DataFrame(y_pred_train_proba)
y_pred_train_proba.to_csv("y-pred-train-celegans-proba-32dim.csv")
   
acc_train = accuracy_score(y_train, y_pred_train)
y_pred_train_proba = pd.read_csv("y-pred-train-celegans-proba-32dim.csv")["1"]
auc_train = roc_auc_score(y_train, y_pred_train_proba)
    
y_pred_test = gc.predict(X_test)  
y_pred_test_proba = gc.predict_proba(X_test) 
y_pred_test = pd.DataFrame(y_pred_test)
#
y_pred_test.to_csv("y-pred-test-celegans-32dim.csv")
y_pred_test = list(pd.read_csv("y-pred-test-celegans-32dim.csv")['0'])
y_pred_test_proba = pd.DataFrame(y_pred_test_proba)
y_pred_test_proba.to_csv("y-pred-test-celegans-proba-32dim.csv")

acc_test = accuracy_score(y_test, y_pred_test)
y_pred_test_proba = list(pd.read_csv("y-pred-test-celegans-proba-32dim.csv")["1"])
auc_test = roc_auc_score(y_test, y_pred_test_proba)

precision_test = precision_score(y_test,y_pred_test)
#recall_score_test = recall_score(y_test,y_pred_test)
f1_score_test = f1_score(y_test,y_pred_test)
matthews_corrcoef_test = matthews_corrcoef(y_test,y_pred_test)
#fpr,tpr,threshold = roc_curve(y_pred_test,y_test)
#SP = 1-fpr[1]

bins = np.array([0, 0.5, 1])
tn, fp, fn, tp = np.histogram2d(y_test, y_pred_test, bins=bins)[0].flatten()
SP=tn/(tn+fp)
SE=tp/(tp+fn)




print("Test Accuracy of GcForest of celegans-train (save and load) = {:.2f} %".format(acc_train * 100))
print("Test Accuracy of GcForest of celegans-test (save and load) = {:.2f} %".format(acc_test * 100))

print("auc of GcForest of celegans-train (save and load) = %.4f" % (auc_train))
print("auc of GcForest of celegans-test (save and load) = %.4f" % (auc_test))

print("precision of celegans-test = %.4f" % (precision_test))
print("recall_score(敏感度(SE)) of celegans-test = %.4f" % (SE))
print("f1_score of celegans-test = %.4f" % (f1_score_test))
print("MCC of celegans-test = %.4f" % (matthews_corrcoef_test))
print("特异性(SP) of celegans-test = %.4f" % (SP))



X_train_enc_test = gc.transform(X_train)
X_test_enc = gc.transform(X_test)
#

X_train_enc_test = X_train_enc_test.reshape((X_train_enc_test.shape[0], -1))
X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
X_train_origin = X_train.reshape((X_train.shape[0], -1))
X_test_origin = X_test.reshape((X_test.shape[0], -1))
#
X_train_enc_test = np.hstack((X_train_origin, X_train_enc_test))
X_test_enc = np.hstack((X_test_origin, X_test_enc))
  
print("X_train_enc_test.shape={}, X_test_enc.shape={}".format(X_train_enc_test.shape,         X_test_enc.shape))

with open("embedding-celegans-32dim-XGB-newmodel.pkl", "rb") as f:
    clf = pickle.load(f)



y_pred_train = clf.predict(X_train_enc_test)
y_pred_train_proba = clf.predict_proba(X_train_enc_test)
y_pred_train = pd.DataFrame(y_pred_train)
y_pred_train_proba = pd.DataFrame(y_pred_train_proba)
y_pred_train_proba.to_csv("y-pred-train-celegans-proba-32dim-newmodel.csv")

acc_train = accuracy_score(y_train, y_pred_train)
y_pred_train_proba = pd.read_csv("y-pred-train-celegans-proba-32dim-newmodel.csv")["1"]
auc_train = roc_auc_score(y_train, y_pred_train_proba)

y_pred_test = clf.predict(X_test_enc)
y_pred_test_proba = clf.predict_proba(X_test_enc) 
y_pred_test = pd.DataFrame(y_pred_test)
#
y_pred_test.to_csv("y-pred-test-celegans-32dim-newmodel.csv")
y_pred_test = list(pd.read_csv("y-pred-test-celegans-32dim-newmodel.csv")['0'])
y_pred_test_proba = pd.DataFrame(y_pred_test_proba)

acc_test = accuracy_score(y_test, y_pred_test)
y_pred_test_proba.to_csv("y-pred-test-celegans-proba-32dim-newmodel.csv")
y_pred_test_proba = list(pd.read_csv("y-pred-test-celegans-proba-32dim-newmodel.csv")["1"])
auc_test = roc_auc_score(y_test, y_pred_test_proba)

precision_test_new = precision_score(y_test,y_pred_test)
#recall_score_test_new = recall_score(y_test,y_pred_test)
f1_score_test_new = f1_score(y_test,y_pred_test)
matthews_corrcoef_test_new = matthews_corrcoef(y_test,y_pred_test)
#fpr,tpr,threshold = roc_curve(y_pred_test,y_test)
#SP_test_new = 1-fpr[1]


bins = np.array([0, 0.5, 1])
tn, fp, fn, tp = np.histogram2d(y_test, y_pred_test, bins=bins)[0].flatten()
SP=tn/(tn+fp)
SE=tp/(tp+fn)




print("Test Accuracy of newmodel of celegans-train (save and load) = {:.2f} %".format(acc_train * 100))
print("Test Accuracy of newmodel of celegans-test (save and load) = {:.2f} %".format(acc_test * 100))

print("auc of newmodel of celegans-train (save and load) = %.4f" % (auc_train))
print("auc of newmodel of celegans-test (save and load) = %.4f" % (auc_test))

print("precision of newmodel of celegans-test = %.4f" % (precision_test_new))
print("recall_score(敏感度(SE)) of newmodel of celegans-test = %.4f" % (SE))
print("f1_score of newmodel of celegans-test = %.4f" % (f1_score_test_new))
print("MCC of newmodel of celegans-test = %.4f" % (matthews_corrcoef_test_new))
print("特异性(SP) of newmodel of celegans-test = %.4f" % (SP))


   


    
 













