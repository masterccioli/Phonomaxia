# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:59:39 2018

@author: maste
"""
from scipy import sparse
import numpy as np
import os
from re import sub
from itertools import permutations

class Phonos:
    
    #comparison: 1 = greater than equal; 0 = less than
    #value: threshold if kind 1; attribute index if kind 0
    #kind: 1 = value, 0 = attribute
    def __init__(self, structure = 'ww', distribution = 'sb', dims = 10, sparsity = .2):
        self.memoryVectors = sparse.csr_matrix((0,0))
        self.dictionary = {}
        if(structure == 'rv'):
            self.structure = 'rv'
            
            if(distribution == 'sb'):
                self.distribution = 'sb'
                self.contextVectors = sparse.csr_matrix((0,dims))
                self.dims = dims
                self.sparsity = sparsity
            elif(distribution == 'gaussian'):
                self.distribution = 'gaussian'
                self.contextVectors = sparse.csr_matrix((0,dims))
                self.dims = dims
        elif (structure == 'wd'):
            self.structure = 'wd'
        elif (structure == 'ww'):
            self.structure = 'ww'
        else:
            pass
            # return an error
            
            
    # External Methods
    #read based on line breaks
    def read(self, pathToCorpus, accumulateFunction, params = None, context = 'par'):
        
        if os.path.isdir(pathToCorpus):
            for u in os.listdir(pathToCorpus):
                corpus = loadCorpus(pathToCorpus + u, context)

                #z = 1
                #read corpus
                for text in corpus:
                    #print(str(z) + "/" + str(len(corpus)))
                    #z = z+1
                    if(len(text) <=1):
                        #print('skipped')
                        continue

                    # ensure word represented in dictionary/context vectors
                    self.updateDict(text)

                    if(params == None):
                        accumulateFunction(self, text)
                    else:
                        accumulateFunction(self, text, params)
        else:
            corpus = loadCorpus(pathToCorpus, context)
            
            z = 1
            #read corpus
            for text in corpus:
                #print(str(z) + "/" + str(len(corpus)))
                #z = z+1
                if(len(text) <=1):
                #    print('skipped')
                    continue

                # ensure word represented in dictionary/context vectors
                self.updateDict(text)

                if(params == None):
                    accumulateFunction(self, text)
                else:
                    accumulateFunction(self, text, params)
        print('Done.')

    def save(self,path,filename):
        import pickle
        with open(path+filename+'.ph', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print('Saved.')

    def load(path, filename):
        import pickle
        # Load object
        with open(path+filename+'.ph', 'rb') as input:
            test1 = pickle.load(input)
        print(filename + ' loaded.')
        return test1
    
    # Internal Methods
    # add word(s) to wXw matrix
    def updateDict(self, words):
        for word in words:
            if(not(word in self.dictionary)):
                # initialize memory vectors with first word
                if(len(self.dictionary) == 0):
                    
                    if(self.structure == 'ww'):
                        self.memoryVectors = sparse.csr_matrix((1,1))
                    
                    elif(self.structure == 'wd'):
                        self.memoryVectors = sparse.csr_matrix((1,0))
                    
                    elif(self.structure == 'rv'):
                        if(self.distribution == 'sb'):
                            #update contextVectors
                            row = np.array(np.repeat(np.arange(1), self.dims*self.sparsity))
                            # no repeating values within a row
                            col = np.random.choice(self.dims,int(self.dims * self.sparsity), replace=False)
                            data = np.repeat(1, len(row))
                            # TODO: ensure unique row vectors
                            self.contextVectors = sparse.csr_matrix((data, (row, col)), shape=(1,self.dims))

                            #update memoryVectors
                            self.memoryVectors = sparse.csr_matrix((1,self.dims))

                        elif(self.distribution == 'gaussian'):
                            
                            #update contextVectors
                            row = np.array(np.repeat(np.arange(1), self.contextVectors.shape[1]))
                            # no repeating values within a row
                            col = np.arange(self.dims)
                            #data = np.random.normal(loc=0,scale=1,size=self.dims)
                            data = np.random.uniform(low=0,high=1,size=self.dims)
                            # TODO: ensure unique row vectors
                            self.contextVectors = sparse.csr_matrix((data, (row, col)), shape=(1,self.dims))

                            #update memoryVectors
                            self.memoryVectors = sparse.csr_matrix((1,self.dims))

                # add word to memoryVectors
                else:
                    
                    if(self.structure == 'ww'):
                        #row
                        self.memoryVectors = sparse.vstack([self.memoryVectors, sparse.csr_matrix((1,self.memoryVectors.shape[1]))],format='csr')
                        #col
                        self.memoryVectors = sparse.hstack([self.memoryVectors, sparse.csr_matrix((self.memoryVectors.shape[0],1))],format='csr')
                    
                    elif(self.structure == 'wd'):
                        row = sparse.csr_matrix((1,self.memoryVectors.shape[1]))
                        self.memoryVectors = sparse.vstack([self.memoryVectors, row],format='csr')
                    
                    elif(self.structure == 'rv'):
                        if(self.distribution == 'sb'):
                            #update contextVectors
                            row = np.zeros(int(self.dims * self.sparsity,),dtype=int)
                            data = np.ones((len(row),),dtype=int)
                            col = np.random.choice(self.dims, int(self.dims * self.sparsity), replace=False)
                            t = sparse.csr_matrix((data, ([row, col])), shape=(1,self.dims))
                            self.contextVectors = sparse.vstack([self.contextVectors,t])

                            #update memoryVectors
                            t = sparse.csr_matrix((1,self.dims))
                            self.memoryVectors = sparse.vstack([self.memoryVectors,t])

                        elif(self.distribution == 'gaussian'):
                            #update contextVectors
                            row = np.array(np.repeat(np.arange(1), self.contextVectors.shape[1]))
                            # no repeating values within a row
                            col = np.arange(self.dims)
                            #data = np.random.normal(loc=0,scale=.5,size=self.dims)
                            data = np.random.uniform(low=0,high=1,size=self.dims)
                            # TODO: ensure unique row vectors
                            t = sparse.csr_matrix((data, (row, col)), shape=(1,self.dims))
                            self.contextVectors = sparse.vstack([self.contextVectors,t])

                            #update memoryVectors
                            t = sparse.csr_matrix((1,self.dims))
                            self.memoryVectors = sparse.vstack([self.memoryVectors,t])

                # set value of word to index of word in new matrix
                self.dictionary[word] = len(self.dictionary)
                
    def getIndices(self, wordList):
        # ensure all words are in dictionary
        wordList = [i for i in wordList if i in self.dictionary.keys()]
        return [self.dictionary[i] for i in wordList]
    
        
class accumulate():
    pass

    # accumulate for a wXw matrix
    def wordByWord(ph, context):
        t = [ph.dictionary[x] for x in context]
        t = list(permutations(t,2))
        t = list(zip(*t))
        data = np.ones((len(t[0]),), dtype=int)
        t = sparse.csr_matrix( (data,(t[0],t[1])), shape=ph.memoryVectors.shape )
        ph.memoryVectors = ph.memoryVectors + t

    # accumulate for a wXd matrix
    def wordByDoc(ph, context):
        t = [ph.dictionary[x] for x in context]
        col = np.zeros((len(t),),dtype=int)
        data = np.ones((len(t),),dtype=int)
        t = sparse.csr_matrix( (data,(t,col)), shape = (len(ph.dictionary),1))
        ph.memoryVectors = sparse.hstack([ph.memoryVectors,t],format='csr')

    # accumulate for random vector matrix
    def randomVector(ph, context):
        t = [ph.dictionary[x] for x in context]
        v = ph.contextVectors[t].nonzero()[1]
        col = np.tile(v,len(t))
        row = np.repeat(t,len(v))
        data = np.ones((len(row),),dtype=int)
        t = sparse.csr_matrix( (data,(row,col)), shape = ph.memoryVectors.shape)

        ph.memoryVectors = ph.memoryVectors + t

        for word in context:
            ph.memoryVectors[ph.dictionary[word]] = ph.memoryVectors[ph.dictionary[word]] - ph.contextVectors[ph.dictionary[word]]

    # the weight of accumulation is inversely proportional to the distance between the words
    # using a fixed-shape window
    def halAccumulator(ph, context, params = [5, False]):
        #print(params[0])
        for i in range(1, params[0]):
            j = 0
            t = []
            while j + i < len(context):
                u = []
                u.append(ph.dictionary[context[j]])
                u.append(ph.dictionary[context[j + i]])
                t.append(u)
                if params[1]:
                    v = []
                    v.append(ph.dictionary[context[j + i]])
                    v.append(ph.dictionary[context[j]])
                    t.append(v)
                j = j + 1
            if(len(t) > 0):
                #data = np.repeat(1 / i, repeats=len(t))
                data = np.repeat(params[0] - i + 1, repeats=len(t))
                t = sparse.csr_matrix( (data, (np.array(t)[:,0], np.array(t)[:,1])), shape = ph.memoryVectors.shape)
                ph.memoryVectors = ph.memoryVectors + t
def loadCorpus(filepath, context = 'paragraph'):
        #load corpus
        corpus = []
        ofile = open(filepath, 'r')
        for line in ofile:
            #remove symbols
            line = sub('\n', '', line)
            line = sub('\'', '', line)
            line = sub('"', '', line)
            line = sub('-', '', line)
            line = sub('\(', '', line)
            line = sub('\)', '', line)
            line = sub(',', '', line)
            line = sub(':', '', line)
            line = line.lower()
            
            if(context == 'sen'):
                line = line.split('.')
                f = line
                line = []
                [line.extend(i.split('?')) for i in f]
                f = line
                line = []
                [line.extend(i.split('!')) for i in f]
                f = line
                line = []
                [line.append(i.split(' ')) for i in f]
                line = [list(filter(None, l)) for l in line]
                line = list(filter(None,line))
                corpus.extend(line)
            # paragraph
            elif(context == 'par'):
                line = sub('\.', '', line)
                line = sub('\?', '', line)
                line = sub('!', '', line)
                line=line.split(' ')
                line = list(filter(None, line))
                corpus.append(line)
            elif(context == 'doc'):
                if(len(corpus) == 0):                
                    line = sub('\.', '', line)
                    line = sub('\?', '', line)
                    line = sub('!', '', line)
                    line=line.split(' ')
                    line = list(filter(None, line))
                    corpus.append(line)
                else:
                    line = sub('\.', '', line)
                    line = sub('\?', '', line)
                    line = sub('!', '', line)
                    line=line.split(' ')
                    line = list(filter(None, line))
                    corpus[0].extend(line)
            # sliding window
            else:
                if(len(corpus) == 0):                
                    line = sub('\.', '', line)
                    line = sub('\?', '', line)
                    line = sub('!', '', line)
                    line=line.split(' ')
                    line = list(filter(None, line))
                    corpus.append(line)
                else:
                    line = sub('\.', '', line)
                    line = sub('\?', '', line)
                    line = sub('!', '', line)
                    line=line.split(' ')
                    line = list(filter(None, line))
                    corpus[0].extend(line)
                
        ofile.close()
        if(isinstance(context,int)):
            f = []
            i = 0
            while(i+context <= len(corpus[0])):
                f.append(corpus[0][i:i+context])
                i = i + 1
            corpus = f
        
        return corpus