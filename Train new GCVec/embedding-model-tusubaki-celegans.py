

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
from PIL import Image




protein_embeddings_train = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[:5222]
length_train = len(protein_embeddings_train["mean_ci_2"])

protein_embeddings_train = [protein_embeddings_train.loc[i].values for i in range(length_train)]


drug_embeddings_train = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[:5222]
drug_embeddings_train = [drug_embeddings_train.loc[i].values for i in range(length_train)]


all_embeddings_train = np.hstack((protein_embeddings_train, drug_embeddings_train))


all_embeddings_train = [all_embeddings_train[i].reshape(64,1) for i in range(length_train)]
 
labels_train = pd.read_csv("celegans-data-pre.csv")["label"][:5222]





protein_embeddings_test = pd.read_csv("celegans-protein-embeddings-32dim.csv").iloc[5222:]
protein_embeddings_test = protein_embeddings_test.reset_index(drop=True)
length_test = len(protein_embeddings_test["mean_ci_2"])
protein_embeddings_test = [protein_embeddings_test.loc[i].values for i in range(length_test)]


drug_embeddings_test = pd.read_csv("celegans-drug-embeddings-32dim.csv").iloc[5222:]
drug_embeddings_test = drug_embeddings_test.reset_index(drop=True)
drug_embeddings_test = [drug_embeddings_test.loc[i].values for i in range(length_test)]


all_embeddings_test = np.hstack((protein_embeddings_test, drug_embeddings_test))
print(type(all_embeddings_test))

all_embeddings_test = [all_embeddings_test[i].reshape(64,1) for i in range(length_test)]
 
labels_test = pd.read_csv("celegans-data-pre.csv")["label"][5222:]


 
X_train = all_embeddings_train
y_train = np.array(labels_train)
X_test = all_embeddings_test
y_test = np.array(labels_test)
print(np.array(X_train).shape)

X_train = np.array(X_train)[:, np.newaxis, :, :]
X_test = np.array(X_test)[:, np.newaxis, :, :]

print(X_test.shape)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        config = get_toy_config()
    else:
        config = load_json(args.model)

    gc = GCForest(config)

    X_train_enc = gc.fit_transform(X_train, y_train)
      

    # dump
    with open("embedding-celegans-32dim-XGB.pkl", "wb") as f:
        pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
    # load
    with open("embedding-celegans-32dim-XGB.pkl", "rb") as f:
        gc = pickle.load(f)
    
    y_pred_train = gc.predict(X_train)    
    acc_train = accuracy_score(y_train, y_pred_train)
    
    y_pred_test = gc.predict(X_test)    
    acc_test = accuracy_score(y_test, y_pred_test)
    print("Test Accuracy of GcForest of celegans-train (save and load) = {:.2f} %".format(acc_train * 100))
    print("Test Accuracy of GcForest of celegans-test (save and load) = {:.2f} %".format(acc_test * 100))



    #passing X_enc to another classfier on top of gcForest.e.g. xgboost/RF.
    X_train_enc_test = gc.transform(X_train)
    X_test_enc = gc.transform(X_test)
    X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
    X_train_enc_test = X_train_enc_test.reshape((X_train_enc_test.shape[0], -1))
    X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
    X_train_origin = X_train.reshape((X_train.shape[0], -1))
    X_test_origin = X_test.reshape((X_test.shape[0], -1))
    X_train_enc = np.hstack((X_train_origin, X_train_enc))
    X_train_enc_test = np.hstack((X_train_origin, X_train_enc_test))
    X_test_enc = np.hstack((X_test_origin, X_test_enc))
  
    print("X_train_enc_test.shape={}, X_test_enc.shape={}".format(X_train_enc_test.shape,         X_test_enc.shape))

    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
    clf.fit(X_train_enc, y_train)
    with open("embedding-celegans-32dim-XGB-newmodel.pkl", "wb") as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    with open("embedding-celegans-32dim-XGB-newmodel.pkl", "rb") as f:
        clf = pickle.load(f)



    y_pred_train = clf.predict(X_train_enc_test)
    acc_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = clf.predict(X_test_enc)
    acc_test = accuracy_score(y_test, y_pred_test)
    print("Test Accuracy of celegans-train of new model using gcforest's X_encode = {:.2f} %".format(acc_train * 100))

    print("Test Accuracy of celegans-test of new model using gcforest's X_encode = {:.2f} %".format(acc_test * 100))


















