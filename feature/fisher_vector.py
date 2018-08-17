import os, sys
import numpy as np
import cv2 as cv
from scipy.stats import multivariate_normal

from util import folder_des
from quantization.gmm import gen_gmm

'''
paper and symbols:
    <Image Classification with the Fisher Vector: Theory and Practice>
'''

def gen_fv(samples, gmm_means, gmm_covs, gmm_weights):

    # Algorithm 1-1 Compute statistics
    s0, s1, s2 = gen_likelihood(samples, gmm_means, gmm_covs, gmm_weights)
    T = samples.shape[0]
    gmm_sigmas = np.float32([np.diagonal(gmm_covs[k]) for k in range(0, gmm_covs.shape[0])])

    # Algorithm 1-2 Compute the Fisher vector signature
    def fv_weight(s0, s1, s2, m, s, w, T):
        return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])

    def fv_means(s0, s1, s2, m, s, w):
        return np.float32([(s1[k] - m[k] * s0[k]) /
            (np.sqrt(w[k] * s[k])) for k in range(0, len(w))])

    def fv_sigma(s0, s1, s2, m, s, w):
        return np.float32([
            (s2[k] - 2 * m[k] * s1[k] + (m[k] * m[k] - s[k]) * s0[k]) / 
            (np.sqrt(2 * w[k]) * s[k]) for k in range(0, len(w))])

    fv_w = fv_weight(s0, s1, s2, gmm_means, gmm_sigmas, gmm_weights, T) # shape: (cluster_K, )
    fv_m = fv_means(s0, s1, s2, gmm_means, gmm_sigmas, gmm_weights)     # shape: (cluster_K, Dimensionality)
    fv_s = fv_sigma(s0, s1, s2, gmm_means, gmm_sigmas, gmm_weights)     # shape: (cluster_K, Dimensionality)
    fv = np.concatenate([fv_w, np.concatenate(fv_m), np.concatenate(fv_s)])

    # Algorithm 1-3 Apply normalizations
    fv = np.sqrt(abs(fv)) * np.sign(fv)
    fv = fv / np.sqrt(np.dot(fv, fv))
    return fv

def gen_likelihood(samples, means, covs, weights):
    samples = zip(range(0, len(samples)), samples)
    g_basis = [multivariate_normal(mean=means[k], cov=covs[k])
            for k in range(0, len(weights))]

    gaussians = {}
    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g_basis])

    s0, s1, s2 = {}, {}, {}
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            ytks = np.multiply(gaussians[index], weights)
            ytks = ytks / np.sum(ytks)
            s0[k] = s0[k] + ytks[k]
            s1[k] = s1[k] + np.power(np.float32(x), 1) * ytks[k]
            s2[k] = s2[k] + np.power(np.float32(x), 2) * ytks[k]
    return s0, s1, s2

def test(folder):
    print "folder: ", folder_des
    # sift
    words = folder_des(folder)
    means, covs, weights = gen_gmm(words, 10)
    fisher_vector = gen_fv(words, means, covs, weights)
    print fisher_vector
