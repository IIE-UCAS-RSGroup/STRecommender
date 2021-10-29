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

def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)

def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)

def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    1/min(m, N) prevents your AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones
    see http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms
    """
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)
    
    
def ndcgk(actual, predicted, k):
    """
    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    see https://github.com/allenjack/HGN/blob/master/eval_metrics.py
    NDCG@k = 1/IDCG sum(2^ri - 1)/log(i+1)
    """
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0 #hit
    for i, p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg
    
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
    
    