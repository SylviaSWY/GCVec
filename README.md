## GCVec
We propose GCVec for the first time. GCVec is an innovative virtual screening model, which utilizes Word2vec (used for natural language processing) to characterize SMILES of drugs and amino acid sequences of targets to generate low-dimensional vectors and multi-grand cascade forest classifier is used to predict whether the drug-target interaction can occur. Compared with DL, the multi-grand cascade forest classifier has the advantages of fewer hyper-parameters and can adaptively set the complexity of the model according to the scale of datasets so as to avoid the over-fitting problem of DL on small-scale datasets. In order to demonstrate the rationality and effectiveness of GCVec, we used the GCVec model to screen small molecule inhibitors of cluster of differentiation 47 (CD47). The results proved that the GCVec model is a powerful tool for the new generation of drug-target binding prediction.

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

## Use guidence
## Prepare environment
```
pip3 install -r requirements.txt
```
## Word2vec to generate low-dimensional vectors of SMILES of drugs and amino acid sequences of proteins
cd the 



## Datasets
Corresponding SMILES sequences are provided in the four directories respectively according to different purposes.
We used [SMILES enumeration](https://github.com/Ebjerrum/SMILES-enumeration) to prepare the sequences.
The sequences are then converted to tokens during data prepocessing. The code used for tokenization is based on the [smiles_tokenizer](https://github.com/topazape/LSTM_Chem/blob/master/lstm_chem/utils/smiles_tokenizer.py) module in LSTM_Chem.  
Before preliminary training, the sequences can be preprocessed by running:
```
python pre_data.py
```
After preparing the preliminary data, the sequences used for transfer learning can be preprocessed by running:
```
python pre_data_tl.py
```

## Basic use
To train the preliminary model:
```
python model.py
```
To perform transfer learning:
```
python tl.py
```
To generate SMILES sequences:
```
python generate.py
```

## Experiments
The SMILES sequences were generated at random, so all the generated sequences were deposited in the four directories according to different purposes.
