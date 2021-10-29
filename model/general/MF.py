# -*- coding: utf-8 -*-
import time
import math
import numpy as np

class MF(object):
    def __init__(self, config):
        self.K = config['dim']
        self.alpha_U = config['alpha_U']
        self.alpha_I = config['alpha_I']
        self.U, self.I = None, None

    def save_model(self, path):
        ctime = time.time()
        print("Saving latent factors U and I...",)
        np.save(path + "U", self.U)
        np.save(path + "I", self.I)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading latent factors U and I...",)
        self.U = np.load(path + "U.npy")
        self.I = np.load(path + "I.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")
        
    def compute_rec_scores(self, sparse_training_matrix, config):
        ctime = time.time()
        print("Training Matrix Factorization model...", )
        K = self.K
        lambda_u = 0.1/self.alpha_U
        lambda_i = 0.1/self.alpha_I

        F = sparse_training_matrix
        M, N = sparse_training_matrix.shape
        U = np.random.normal(0, self.alpha_U, (M, K))
        I = np.random.normal(0, self.alpha_I, (N, K))
        
        F = F.tocoo() 
        entry_index = list(zip(F.row, F.col))
        F = F.tocsr() 
        F_dok = F.todok()

        # tau = 10
        last_loss = float('Inf')
        for iters in range(config['iters_num']):
            F_Y = F_dok.copy()
            for i, j in entry_index:
                F_Y[i, j] = F_dok[i, j] - U[i].dot(I[j])
            F_Y = F_Y.tocsr()

            learning_rate_k = config['lr']
            U += learning_rate_k * (F_Y.dot(I) - lambda_i * U)
            I += learning_rate_k * ((F_Y.T).dot(U) - lambda_u * I)

            loss = 0.0
            for i, j in entry_index:
                loss += 0.5 * ((F_dok[i, j] - U[i].dot(I[j]))**2 + lambda_u * np.square(U[i]).sum() + lambda_i * np.square(I[j]).sum())

            print('Iteration:', iters,  'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.U, self.I = U, I


    def predict(self, u, p, sigmoid = False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[u].dot(self.I[p])))
        return self.U[u].dot(self.I[p])
