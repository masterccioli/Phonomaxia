# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:36:55 2018

@author: maste
"""
import numpy as np
import pandas as pd

class sim():
    # this is a class for methods processing similarities
    # see Bullinaria&Levy for measures
    # returns a similarity matrix (need a way to subset the dictionary/memory matrices)
    # e.g. cosine, euclidean
    
    # cosine distance matrix
    def cosine(ph, words):
        num = ph.memoryVectors[ph.getIndices(words)].dot(ph.memoryVectors[ph.getIndices(words)].transpose())
        denom = np.multiply(np.tile((np.sqrt((ph.memoryVectors[ph.getIndices(words)]).power(2).sum(1))),ph.memoryVectors[ph.getIndices(words)].shape[0]), np.tile((np.sqrt((ph.memoryVectors[ph.getIndices(words)]).power(2).sum(1))),ph.memoryVectors[ph.getIndices(words)].shape[0]).transpose())
        return [pd.DataFrame(data = num/denom, index = words, columns = words), num/denom]
                                                                        
    # euclidean distance matrix
    def euclidean(ph,words):
        e = np.empty([ph.memoryVectors[ph.getIndices(words)].shape[0], ph.memoryVectors[ph.getIndices(words)].shape[0]])
        for w in np.arange(ph.memoryVectors[ph.getIndices(words)].shape[0]):
            e[w] = np.sqrt(np.power(ph.memoryVectors[ph.getIndices(words)] - np.tile(ph.memoryVectors[ph.getIndices(words)][w].toarray(),reps=(ph.memoryVectors[ph.getIndices(words)].shape[0],1)),2).sum(1)).transpose()
        return pd.DataFrame(data = e, index = words, columns = words)
		
    #def singleCosine(ph,memoryVectors,target):
    def singleCosine(ph,words,target):
        num = ph.memoryVectors[ph.getIndices(words)].dot(ph.memoryVectors[ph.getIndices(target)].transpose())
        denom = np.multiply((np.sqrt((ph.memoryVectors[ph.getIndices(words)]).power(2).sum(1))), (np.sqrt((ph.memoryVectors[ph.getIndices(target)]).power(2).sum(1))))
        return pd.DataFrame(data = num/denom, index = words, columns = [target])
    
class vis():
    def mdsPlot(ph, words):
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection
        
        from sklearn import manifold
        from sklearn.metrics import euclidean_distances
        from sklearn.decomposition import PCA
        
        seed = np.random.RandomState(seed=3)
        
        similarities = 1 - sim.cosine(ph, words)[1]
        
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(similarities).embedding_
        
        # Rotate the data
        clf = PCA(n_components=2)
        pos = clf.fit_transform(pos)
        
        s = 100
        
        plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
        plt.rc('font', size=20)
        
        similarities = similarities.max() / similarities * 100
        similarities[np.isinf(similarities)] = 0
        
        for i, txt in enumerate(words):
            plt.annotate(txt, (pos[i,0],pos[i,1]))
        
        plt.show()