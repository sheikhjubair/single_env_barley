import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class hw_encoder:
    def __init__(self, homoGeno_1, homoGeno_2, heteroGeno):
        self.homoGeno_1 = homoGeno_1
        self.homoGeno_2 = homoGeno_2
        self.heteroGeno = heteroGeno
        
    def fit(self, germplasm):
        d = germplasm == self.homoGeno_1
        d = d.sum()
        h = germplasm == self.heteroGeno
        h = h.sum()

        self.p = (2 * d + h) / (2 * len(germplasm))
        self.q = 1 - self.p
        self._2pq = 2 * self.p * self.q
        self.p_square = self.p ** 2
        self.q_square = self.q ** 2
        
    def transform(self, germplasm):
        for col in germplasm:
            germplasm[col] = germplasm[col].replace(self.homoGeno_1, self.p_square[col])
            germplasm[col] = germplasm[col].replace(self.homoGeno_2, self.q_square[col])
            germplasm[col] = germplasm[col].replace(self.heteroGeno, self._2pq[col])
            
        return germplasm
    
    def fit_transform(self, germplasm):
        self.fit(germplasm)
        return self.transform(germplasm)
        
        
# def encode_hardy_weinberg(germplasm, homoGeno_1, homoGeno_2, heteroGeno):
#     d = germplasm == homoGeno_1
#     d = d.sum()
#     h = germplasm == heteroGeno
#     h = h.sum()

#     p = (2 * d + h) / (2 * len(germplasm))
#     q = 1 - p
#     _2pq = 2 * p * q
#     p_square = p ** 2
#     q_square = q ** 2

#     for col in germplasm:
#         germplasm[col] = germplasm[col].replace(homoGeno_1, p_square[col])
#         germplasm[col] = germplasm[col].replace(homoGeno_2, q_square[col])
#         germplasm[col] = germplasm[col].replace(heteroGeno, _2pq[col])

#     return germplasm

def encode_one_hot(germplasm, categories):
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(np.array(categories).reshape(-1, 1))
    cols = germplasm.columns
    num_row, num_col = germplasm.shape
    germplasm = germplasm.to_numpy().reshape(-1, 1)
    encoded = enc.transform(germplasm)
    encoded = np.reshape(encoded,(num_row, num_col, 3))
    
    return encoded


class simple_encoder:
    def __init__(self, genotypes, geno_reps):
        """
            genotypes = a list of n unique genotypes in the germplasm. 
            geno_reps = a list of representation of each genotypes. size is n.
                Each element should be an int
        
        """
        
        self.genotypes = genotypes
        self.geno_reps = geno_reps
        

    def fit_transform(self, germplasm):
        """
            germplasm = a m x n dimensional numpy array
        """
        encoded = np.zeros(germplasm.shape, dtype = np.int)
        for i, geno in enumerate(self.genotypes):
            ind = germplasm == geno
            encoded[ind] = self.geno_reps[i]
        
            
        return encoded