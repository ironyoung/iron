import os, sys
import numpy as np
import cv2 as cv
from util import folder_des

def gen_gmm(words, cluster_K):
    em = cv.ml.EM_create()
    em.setClustersNumber(cluster_K)
    em.trainEM(words)
    means   = np.float32(em.getMeans())
    covs    = np.float32(np.array(em.getCovs()))    # list of ndarray -> ndarray
    weights = np.float32(em.getWeights()[0])        # (1,cluster_K) -> (cluster_K, )

    '''
    #Throw away gaussians with weights that are too small:
    th = 1.0 / cluster_K
    means   = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
    weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])
    covs    = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
    '''
    return means, covs, weights

def test(folder):
    print "folder: ", folder_des
    # sift
    words = folder_des(folder)
    a,b,c = gen_gmm(words, 10)
    print a.shape
    print b.shape
    print c.shape
