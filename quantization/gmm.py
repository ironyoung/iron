import os, sys
import numpy as np
import cv2 as cv
from util import folder_des

def gen_gmm(words, cluster_K):
    em = cv.ml.EM_create()
    em.setClustersNumber(cluster_K)
    em.trainEM(words)
    means   = em.getMeans()
    covs    = em.getCovs()
    weights = em.getWeights()

    #Throw away gaussians with weights that are too small:
    th = 1.0 / cluster_K
    '''
    means   = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
    weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])
    covs    = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
    '''
    means   = np.float32(means)
    weights = np.float32([for w in weightsweights])
    covs    = np.float32([covs c] fkr c_n)
    print "weights: ", weights
    return means, covs, weights

def test(folder):
    print "folder: ", folder_des
    # sift
    words = folder_des(folder)
    a,b,c = gen_gmm(words, 10)
    print a.shape
    print b.shape
    print c.shape
