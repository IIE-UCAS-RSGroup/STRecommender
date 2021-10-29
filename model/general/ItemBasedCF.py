# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
import time

np.set_printoptions(threshold=np.inf)

class ItemBasedCF(object):
    def __init__(self, config):
        self.rec_score = None

    def load_result(self, path):
        ctime = time.time()
        print("Loading recommendation scores...",)
        self.rec_score = np.load(path + "ibcf_rec_score.npy")
        print("Done! Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving recommendation scores...",)
        np.save(path + "ibcf_rec_score", self.rec_score)
        print("Done! Elapsed time:", time.time() - ctime, "s")

    def compute_rec_scores(self, training_matrix, config):
        ctime = time.time()
        print("Training Item-based Collaborative Filtering...", )
        
        sim = training_matrix.T.dot(training_matrix)
        norms = [norm(training_matrix.T[i]) for i in range(training_matrix.shape[1])] 

        for i in range(training_matrix.shape[1]):
            sim[i][i] = 0.0
            for j in range(i + 1, training_matrix.shape[1]):
                sim[i][j] /= (norms[i] * norms[j] + 0.000001)
                sim[j][i] /= (norms[i] * norms[j] + 0.000001)

        if config['select_item_num'] <= sim.shape[1]:
            sim_ = sim * (np.argsort(np.argsort(sim)) >= sim.shape[1] - config['select_item_num'])
            self.rec_score = training_matrix.dot(sim_.T)
        else:
            print("Number of similar users selected is too large")
            self.rec_score = training_matrix.dot(sim)

        # self.rec_score = training_matrix.dot(sim)
        print("Done! Elapsed time:", time.time() - ctime, "s")

    def predict(self, u, p):
        return self.rec_score[u][p]
