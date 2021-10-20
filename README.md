## GCVec
We propose GCVec for the first time. GCVec is an innovative virtual screening model, which utilizes Word2vec (used for natural language processing) to characterize SMILES of drugs and amino acid sequences of targets to generate low-dimensional vectors and multi-grand cascade forest classifier is used to predict whether the drug-target interaction can occur. Compared with DL, the multi-grand cascade forest classifier has the advantages of fewer hyper-parameters and can adaptively set the complexity of the model according to the scale of datasets so as to avoid the over-fitting problem of DL on small-scale datasets. In order to demonstrate the rationality and effectiveness of GCVec, we used the GCVec model to screen small molecule inhibitors of cluster of differentiation 47 (CD47). The results proved that the GCVec model is a powerful tool for the new generation of drug-target binding prediction. GCVec was constructed based on the innitial gcforest model constructed by Zhou, Z[1] with several changes and modifications. And the part that Word2vec to generate low-dimensional vectors of SMILES of drugs and amino acid sequences of proteins could be refered to in Yu Fang Zhang's[2] work with modifications.

## Requirements:
* argparse
* joblib
* keras
* psutil
* scikit-learn>=0.18.1
* scipy
* simplejson
* tensorflow
* xgboost

## User guidence
## Prepare environment
```
pip3 install -r requirements.txt
```
## Word2vec to generate low-dimensional vectors of SMILES of drugs and amino acid sequences of proteins
(to generate the low-dimensional vectors of celegans dataset as an example)
```
cd  Word2vec
python predata.py
```
## Train new GCVec 
(to train the GCVec model using celegans dataset as an example)
```
python ./Train new GCVec/embedding-model-tusubaki-celegans.py --model ./Train new GCVec/embedding-32dim-XGB.json
```
## Trained GCVec application
After taining the new GCVec, it will generate two trained model files, in this example, the generated two trained model files are "embedding-celegans-32dim-XGB.pkl" and "embedding-celegans-32dim-XGB-newmodel.pkl", copy these two files to ./Trained GCVec application, then you can utilize the trained GCVec to predict new drug-target interactions, in this example, the trained GCVec model will be used to predict the test set of celegans dataset.
```
cd ./Trained GCVec application
python embedding-celegans-test.py
```
## Other sections in the paper
The process of other section in this paper follow the above process: Word2vec to generate vectors for drugs and proteins ---> Train new GCVec with the prepared Datasets ---> Trained GCVec to predict new drug-target interactions. 

## Supplementary notes for all Datasets
All datasets concerned in this article can be downloaded from zenodo website "https://doi.org/10.5281/zenodo.5584700", the downloaded file contains all raw data and processed data concerned in the article "GCVec: a new screening model towards the CD47 inhibitors exploration", this file contain 6 sub-files, namely GCVec——humans, GCVec——celegans, GCVec——challenging dataset, GCVec——blindly screen CD47, DS-docking CD47, GCVec——specs, respectively, fully cover all parts of the article.


## References
[1] Zhou, Z.; Feng, J. In Deep forest: towards an alternative to deep neural networks, International Joint Conference on Artificial Intelligence, 2017; pp 3553-3559.

[2] Zhang, Y. F.;  Wang, X.;  Kaushik, A. C.;  Chu, Y.;  Shan, X.;  Zhao, M. Z.;  Xu, Q.; Wei, D. Q., SPVec: A Word2vec-Inspired Feature Representation Method for Drug-Target Interaction Prediction. Front Chem 2019, 7, 895.
