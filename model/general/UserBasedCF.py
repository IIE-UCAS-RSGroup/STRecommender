# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
import time

class UserBasedCF(object):
    def __init__(self, config):
        self.rec_score = None

    def load_result(self, path):
        ctime = time.time()
        print("Loading recommendation scores...",)
        self.rec_score = np.load(path + "ubcf_rec_score.npy")
        print("Done! Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving recommendation scores...",)
        np.save(path + "ubcf_rec_score", self.rec_score)
        print("Done! Elapsed time:", time.time() - ctime, "s")

    def compute_rec_scores(self, training_matrix, config):
        ctime = time.time()
        print("Training User-based Collaborative Filtering...", )

        sim = training_matrix.dot(training_matrix.T)
        norms = [norm(training_matrix[i]) for i in range(training_matrix.shape[0])] 

        for i in range(training_matrix.shape[0]):
            sim[i][i] = 0.0
            for j in range(i + 1, training_matrix.shape[0]):
                sim[i][j] /= (norms[i] * norms[j] + 0.000001)
                sim[j][i] /= (norms[i] * norms[j] + 0.000001)

        if config['select_user_num'] <= sim.shape[1]:
            sim_ = sim * (np.argsort(np.argsort(sim)) >= sim.shape[1] - config['select_user_num'])
            self.rec_score = sim_.dot(training_matrix)
        else:
            print("Number of similar users selected is too large")
            self.rec_score = sim.dot(training_matrix)

        print("Done! Elapsed time:", time.time() - ctime, "s")

    def predict(self, u, p):
        return self.rec_score[u][p]
