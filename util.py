import os, sys
import numpy as np
import cv2 as cv
import math
import argparse, glob, time

def img_des(img_file, flag = 0):
    img = cv.imread(img_file, flag)
    img = cv.resize(img, (256, 256))
    sift = cv.xfeatures2d.SIFT_create()
    #kp, des = cv.SIFT().detectAndCompute(img, None)
    kp, des = sift.detectAndCompute(img, None)
    return des

def folder_des(folder):
    files = glob.glob(folder + '/*.jpg')
    return np.concatenate([img_des(f) for f in files])

if __name__ == '__main__':
    folder = sys.argv[1]
    print "folder: ", folder
    print folder_des(folder).shape
