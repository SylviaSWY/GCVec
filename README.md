# GCVec
We propose GCVec for the first time. GCVec is an innovative virtual screening model, which utilizes Word2vec (used for natural language processing) to characterize SMILES of drugs and amino acid sequences of targets to generate low-dimensional vectors and multi-grand cascade forest classifier is used to predict whether the drug-target interaction can occur. Compared with DL, the multi-grand cascade forest classifier has the advantages of fewer hyper-parameters and can adaptively set the complexity of the model according to the scale of datasets so as to avoid the over-fitting problem of DL on small-scale datasets. In order to demonstrate the rationality and effectiveness of GCVec, we used the GCVec model to screen small molecule inhibitors of cluster of differentiation 47 (CD47). The results proved that the GCVec model is a powerful tool for the new generation of drug-target binding prediction.

#Environment
argparse
joblib
keras
psutil
scikit-learn>=0.18.1
scipy
simplejson
tensorflow
xgboost

#Use guidence
#Prepare environment
