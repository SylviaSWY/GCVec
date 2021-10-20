import pandas as pd
import numpy as np
import gensim 
import jieba
from gensim.models import Word2Vec 
import random 
from sklearn.decomposition import IncrementalPCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import os 
import random
import seaborn as sns 
import tempfile


protein_seq = pd.read_csv('celegans-data-pre.csv', sep=",")['proseq']
drug_Smi = pd.read_csv('celegans-data-pre.csv', sep=",")['smiles']
print(len(drug_Smi))
class Word2vec:
    def protein2vec(dims,window_size,negative_size):
        word_vec = pd.DataFrame()
        dictionary=[]
        Index = []
        #data=self.read_data()
        texts = [[word for word in re.findall(r'.{3}',str(document))] for document in list(protein_seq)]
        print(texts)
        new_model = gensim.models.Word2Vec.load('gensim-model-32dim-bindingDBchembl-protein-pvnepo_v')
        print(new_model)
        word = new_model.wv.index_to_key
        vector = [new_model.wv.get_vector(a) for a in word]
        vectors = pd.DataFrame(vector)
        #vectors = pd.DataFrame([new_model.wv.get_vector(word for word in new_model.wv.index_to_key)])
        vectors['Word'] = list(new_model.wv.index_to_key)
        print(vectors)  

        for i in range(len(protein_seq)):
            Index.append(i)
        # Word segmentation
        for i in range(len(texts)):
            i_word=[]         
            for w in range(len(texts[i])):
                i_word.append(Index[i])    
            dictionary.extend(i_word)
        word_vec['Id'] = dictionary
        
        # word vectors generation
        dictionary=[]
        for i in range(len(texts)):
            i_word=[]         
            for w in range(len(texts[i])):
                i_word.append(texts[i][w])    
            dictionary.extend(i_word)
        word_vec['Word'] = dictionary
        del dictionary,i_word
        word_vec = word_vec.merge(vectors,on='Word', how='left')
        #word_vec = word_vec.drop('Word',axis=1)
        word_vec.columns = ['Id']+['word']+["vec_{0}".format(i) for i in range(0,dims)]
        print(word_vec)
        return word_vec

    def smiles2vec(dims,window_size,negative_size):
        word_vec = pd.DataFrame()
        dictionary=[]
        Index = []
        #data=self.read_data()
        texts = [[word for word in re.findall(r'.{3}',str(document))] for document in list(drug_Smi)]
        print(texts)
        new_model = gensim.models.Word2Vec.load('gensim-model-32dim-bindingDBchembl-smiles-twu8an90')
        print(new_model)
        word = new_model.wv.index_to_key
        vector = [new_model.wv.get_vector(a) for a in word]
        vectors = pd.DataFrame(vector)
        #vectors = pd.DataFrame([new_model.wv.get_vector(word for word in new_model.wv.index_to_key)])
        vectors['Word'] = list(new_model.wv.index_to_key)
        print(vectors)

        for i in range(len(drug_Smi)):
            Index.append(i)
        # Word segmentation
        for i in range(len(texts)):
            i_word=[]         
            for w in range(len(texts[i])):
                i_word.append(Index[i])    
            dictionary.extend(i_word)
        word_vec['Id'] = dictionary
        
        # word vectors generation
        dictionary=[]
        for i in range(len(texts)):
            i_word=[]         
            for w in range(len(texts[i])):
                i_word.append(texts[i][w])    
            dictionary.extend(i_word)
        word_vec['Word'] = dictionary
        del dictionary,i_word
        word_vec = word_vec.merge(vectors,on='Word', how='left')
        #word_vec = word_vec.drop('Word',axis=1)
        word_vec.columns = ['Id']+['word']+["vec_{0}".format(i) for i in range(0,dims)]
        print(word_vec)
        return word_vec


    #Molecular Structure and Protein Sequence Representation
    def feature_embeddings_protein(dims):
        protein_vec = Word2vec.protein2vec(dims,12,15)
        protein_vec=protein_vec.drop('word',axis=1)
        name = ["vec_{0}".format(i) for i in range(0,dims)]
        feature_embeddings = pd.DataFrame(protein_vec.groupby(['Id'])[name].agg('mean')).reset_index()
        feature_embeddings.columns=["Index"]+["mean_ci_{0}".format(i) for i in range(0,dims)]
        return feature_embeddings


    def feature_embeddings_smiles(dims):
        smiles_vec = Word2vec.smiles2vec(dims,12,15)
        smiles_vec=smiles_vec.drop('word',axis=1)
        name = ["vec_{0}".format(i) for i in range(0,dims)]
        feature_embeddings = pd.DataFrame(smiles_vec.groupby(['Id'])[name].agg('mean')).reset_index()
        feature_embeddings.columns=["Index"]+["mean_ci_{0}".format(i) for i in range(0,dims)]
        return feature_embeddings



if __name__=='__main__':
    print ("Molecular Structure and Protein Sequence Continuous Representation")
    print ("*********************************************")
    try:
        prot_embeddings = Word2vec.feature_embeddings_protein(32)
        drug_embeddings = Word2vec.feature_embeddings_smiles(32)
        #prot_embeddings['proteinseq']=protein_seq
        #drug_embeddings['smiles']=drug_Smi
        del drug_embeddings["Index"]
        del prot_embeddings["Index"]
        prot_embeddings.to_csv('celegans-protein-embeddings-32dim.csv',index=False,sep='\t')
        drug_embeddings.to_csv('celegans-drug-embeddings-32dim.csv',index=False,sep='\t')
    except ImportError:
        print ('Molecular Structure and Protein Sequence Continuous Representation error! ')

    finally:
        print ('******Molecular Structure and Protein Sequence Continuous Representation finished!*********')

