# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:35:46 2018

@author: maste
"""
from scipy import sparse
import numpy as np

class lateAssFunctions():
    pass
    # this is a class for methods of processing matrices
    # e.g. tf-idf
    
    # term frequency (tf) transformations
    
    # binary - 1 if greater than 0, else 0
    def binary(sv):
        u = sparse.find(sv)
        return sparse.csr_matrix((np.repeat(1, len(u[0])), (u[0], u[1])))
    
    # log normalization
    # log(1 + x) rather than log(x) for instances where term doesn't occur
    def logNorm(sv):
        return sparse.csr_matrix(1 + sv.log1p().toarray())
    
    # double normalization
    # K defaults to .5
    # 0 <= K <= 1
    def doubleNorm(sv, k = .5):
        return sparse.csr_matrix(k + k * sv / np.tile(sv.max(axis=0).toarray()[0],reps=(sv.shape[0],1)))

    def logAverage(sv):
        return (np.ones(sv.shape) + sv.log1p()) / (np.ones(sv.shape) + sparse.csr_matrix(np.tile(sv.sum(axis = 0)/sv.shape[0], reps=(sv.shape[0],1))).log1p())
    
    
    # document frequency transformations
    # note shape of idf transformations (t x 1)
    
    # inverse document frequency
    # ** only does idf smooth
    def idf(sv):
        u = sparse.find(sv)
        u = sparse.csr_matrix((np.repeat(1, len(u[0])), (u[0], u[1]))).sum(axis=1)
        return sparse.csr_matrix(np.tile(sv.shape[1],reps=u.shape) / u).log1p()
    
    # max here taken to mean the largest number of documents that a single term appears in
    def idfMax(sv):
        u = sparse.find(sv)
        u = sparse.csr_matrix((np.repeat(1, len(u[0])), (u[0], u[1]))).sum(axis=1)
        return sparse.csr_matrix(np.tile(u.max(),reps = u.shape) / u).log1p()
    
    def idfProb(sv):
        u = sparse.find(sv)
        u = sparse.csr_matrix((np.repeat(1, len(u[0])), (u[0], u[1]))).sum(axis=1)
        return sparse.csr_matrix((np.tile(sv.shape[1],reps=u.shape) - u) / u).log1p()
    
	
	# LSA processing
    def lsaPreprocess(sv):
        sv1 = sv.toarray() + 1
        t = np.divide(sv1, np.tile(sv1.sum(axis = 1), reps=[sv1.shape[1],1]).transpose())
        entropy = -(t * np.log(t)).sum(axis = 1)
        return np.divide(sv1, np.tile(entropy, reps=[sv1.shape[1],1]).transpose())

    def svdReduction(sv, dims):
        u, s, v = np.linalg.svd(sv.toarray(),full_matrices=False, compute_uv=True)
        return np.dot(u[:,:dims], np.diag(s[:dims]))
    
	# Gradient descent function
    # GloVe latent association function - iterate by non-zero value in sparse matrix
    def glove(sv, dims = 50, learningRate = 0.05, iterations = 10, returnCost = False):
        #initialize memory and context vectors
        #random uniform -.5/dims <= x <= .5/dims
        
        # initialize memory and context vectors
        # random uniform -.5/dims <= x <= .5/dims
        memvect = np.random.rand(sv.shape[0], dims) - .5
        contvect = np.random.rand(sv.shape[0], dims) - .5
        biasm = np.random.rand(sv.shape[0]) - .5
        biasc = np.random.rand(sv.shape[0]) - .5
    
        # initialize gradient descent vectors
        gradMem = np.ones(shape=(sv.shape[0],dims))
        gradBiasMem = np.ones(sv.shape[0])
        gradCont = np.ones(shape=(sv.shape[0],dims))
        gradBiasCont = np.ones(sv.shape[0])
    
        cost = np.zeros(iterations)
    
        for i in range(0, iterations):
        #for each non empty entry in the accumulation matrix
            for j in np.random.permutation(range(0,len(list(sv.nonzero())[0]))):
                n = list(sv.nonzero())[0][j]
                v = list(sv.nonzero())[1][j]
                #for v in range(0,len(memvect)):
                # calculate cost
                diff = np.dot(memvect[n], contvect[v].transpose()) + biasm[n] + biasc[v] - np.log(sv[n,v])
                weightedDiff = (sv[n,v] / 100) * diff
                cost[i] = cost[i] + .5 * np.sum(np.multiply(diff, weightedDiff))
    
                # weighted updates
                u = weightedDiff * learningRate
    
                # update memory and context vectors
                memN = u * contvect[v]
                memV = u * memvect[n]
    
                updateN = memN / np.sqrt(gradMem[n])
                updateV = memV / np.sqrt(gradCont[v])
    
                memvect[n] = memvect[n] - updateN
                contvect[v] = contvect[v] - updateV
    
                #update gradient vectors
                gradMem[n] = gradMem[n] + np.multiply(memN, memN)
                gradCont[v] = gradCont[v] + np.multiply(memV, memV)
    
                # update bias vectors
                biasm[n] = biasm[n] - u / np.sqrt(gradBiasMem[n])
                biasc[v] = biasc[v] - u / np.sqrt(gradBiasCont[v])
    
                # update gradient bias vectors
                gradBiasMem[n] = gradBiasMem[n] + np.power(u,2)
                gradBiasCont[v] = gradBiasCont[v] + np.power(u,2)
                
        if(returnCost):
            return memvect,cost
        else:
            return memvect

class vectorNormalization():
    # Vector Normalization Procedures
    def correlation(sparseMatrix):
        
        # correlation function
        # transforms input values to their correlational values, defined in Rhode, Gonnerman, Plaut 2006
    
        T = sparseMatrix.sum()
        beta = sparseMatrix.sum(axis=1)
        gamma = sparseMatrix.sum(axis = 0)
        num = T * sparseMatrix - np.dot(beta,gamma)
        denom = np.sqrt(np.dot(np.multiply(beta,(T - beta)), np.multiply(gamma, T - gamma)))
        denom[[index for index, i in enumerate(beta) if i == 0],:] = -.1 # set zero vals to small negative number to prevent divide errors
        out = np.divide(num,denom)
        negs = np.where(out < 0)
        out[negs[0],negs[1]] = 0
        out = np.sqrt(out)
        out = sparse.csr_matrix(out)
        return out
        #print(num / denom)
    
            #if(i == 0):
                #print(index)
    def row(sparseMatrix):
        return sparse.csr_matrix(sparseMatrix / sparseMatrix.sum(axis=1))

    def column(sparseMatrix):
        return sparse.csr_matrix(sparseMatrix / sparseMatrix.sum(axis=0))

class operations():
    #get following given vector
    
    def conv_circ( signal, ker ):
        from numpy.fft import fft, ifft
        import numpy as np
        '''
            signal: real 1D array
            ker: real 1D array
            signal and ker must have same shape
        '''
        return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

    def periodic_corr(x, y):
        from numpy.fft import fft, ifft
        import numpy as np
        """Periodic correlation, implemented using the FFT.

        x and y must be real sequences with the same length.
        remove y from x
        """
        return ifft(fft(x) * fft(y).conj()).real