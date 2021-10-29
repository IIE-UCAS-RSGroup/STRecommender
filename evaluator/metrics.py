# -*- coding: utf-8 -*-
import numpy as np
import math

"""
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
"""
class precisionN(object):
    def __init__(self, config):
        self.top_n_list = config['top_n_list']
        self.precisions = []

    def compute(self, actual, predict):
        for i in range(len(self.top_n_list)):
            predicted = predict[: self.top_n_list[i]]
            if len(self.precisions) < len(self.top_n_list):
                temp = []
                temp.append(1.0 * len(set(actual) & set(predicted)) / len(predicted))
                self.precisions.append(temp)
            else:
                self.precisions[i].append(1.0 * len(set(actual) & set(predicted)) / len(predicted))

    def result(self):
        return np.mean(self.precisions, axis = 1)

class recallN(object):
    def __init__(self, config):
        self.top_n_list = config['top_n_list']
        self.recalls = []

    def compute(self, actual, predict):
        for i in range(len(self.top_n_list)):
            predicted = predict[: self.top_n_list[i]]
            if len(self.recalls) < len(self.top_n_list):
                temp = []
                temp.append(1.0 * len(set(actual) & set(predicted)) / len(actual))
                self.recalls.append(temp)
            else:
                self.recalls[i].append(1.0 * len(set(actual) & set(predicted)) / len(actual))

    def result(self):
        return np.mean(self.recalls, axis = 1)

class mapN(object):
    def __init__(self, config):
        self.top_n_list = config['top_n_list']
        self.maps = []

    def compute_apN(self, actual, predict, topN):
        hits = 0
        avg_p = 0

        seen = set()
        for i, recommendation in enumerate(predict[:topN]):
            if recommendation in actual and recommendation not in seen:
                hits += 1
                avg_p += hits / float(i + 1)
            seen.add(recommendation)
        return avg_p / min(topN, len(actual))

    def compute(self, actual, predict):
        for i in range(len(self.top_n_list)):
            if len(self.maps) < len(self.top_n_list):
                temp = []
                temp.append(np.mean([self.compute_apN(actual, predict, self.top_n_list[i])]))
                self.maps.append(temp)
            else:
                self.maps[i].append(np.mean([self.compute_apN(actual, predict, self.top_n_list[i])]))

        if not actual:
            return 0.0

    def result(self):
        return np.mean(self.maps, axis = 1)

class ndcgN(object):
    def __init__(self, config):
        self.top_n_list = config['top_n_list']
        self.ndcgs = []

    def _is_intersection(self, list1, list2):
        return len(set(list1).intersection(list2)) > 0

    def compute_ndcg(self, actual, predict, topN):
        dcg_n = 0.0
        idcg_n = 0.0
        for j in range(min(len(predict), topN)):
            idcg_n += 1. / math.log2(j + 2)
            if self._is_intersection(actual, [predict[j]]):
                dcg_n += 1. / math.log2(j + 2)
        ndcg_n = dcg_n * 1. / idcg_n

        return  ndcg_n

    def compute(self, actual, predict):
        for i in range(len(self.top_n_list)):
            if len(self.ndcgs) < len(self.top_n_list):
                temp = []
                temp.append(self.compute_ndcg(actual, predict, self.top_n_list[i]))
                self.ndcgs.append(temp)
            else:
                self.ndcgs[i].append(self.compute_ndcg(actual, predict, self.top_n_list[i]))

    def result(self):
        return np.mean(self.ndcgs, axis = 1)

# def precisionk(actual, predicted):
#     return 1.0 * len(set(actual) & set(predicted)) / len(predicted)
#
# def recallk(actual, predicted):
#     return 1.0 * len(set(actual) & set(predicted)) / len(actual)
#
# def mapk(actual, predicted, k):
#     """
#     Computes the mean average precision at k.
#     1/min(m, N) prevents your AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones
#     see http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms
#     """
#     score = 0.0
#     num_hits = 0.0
#
#     for i, p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)
#
#     if not actual:
#         return 0.0
#
#     return score / min(len(actual), k)
#
#
# def ndcgk(actual, predicted, k):
#     """
#     Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
#     see https://github.com/allenjack/HGN/blob/master/eval_metrics.py
#     NDCG@k = 1/IDCG sum(2^ri - 1)/log(i+1)
#     """
#     idcg = 1.0
#     dcg = 1.0 if predicted[0] in actual else 0.0 #hit
#     for i, p in enumerate(predicted[1:]):
#         if p in actual:
#             dcg += 1.0 / np.log(i+2)
#         idcg += 1.0 / np.log(i+2)
#     return dcg / idcg
    
def hitratek(actual, predicted):
    """
       HR@K = SUM(Number of Hits @K)/SUM(actual)
    """
    hits = 1.0 * len(set(actual) & set(predicted))
    gts = len(actual)
    return hits, gts

# if __name__ == '__main__':
#     def testhitratek(actual, predicted):
#         hit = []
#         gt = []
#         for cnt, uid in enumerate(actual):
#             hits, gts = hitratek(actual[cnt], predicted[cnt])
#             hit.append(hits)
#             gt.append(gts)
#         print("HitRate:", np.sum(hit)/np.sum(gt))
        
#     def testndcg(actual, predicted):
#         s1 = []
#         s2 = []
#         for cnt, uid in enumerate(actual):
#             nd1 = ndcg(actual[cnt], predicted[cnt],2)
#             nd2 = ndcgk(actual[cnt], predicted[cnt],2)
#             s1.append(nd1)
#             s2.append(nd2)
#         print("ndcg:", np.mean(s1))
#         print("ndcgk:", np.mean(s2))
#     actual = [[1, 2], [3, 4, 5]]
#     predicted = [[10, 20, 1, 30, 40], [10, 3, 20, 4, 5]]
#     testndcg(actual, predicted)
    
    