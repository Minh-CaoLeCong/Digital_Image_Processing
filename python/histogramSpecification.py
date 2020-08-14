# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python histogramSpecification.py --input imagePath
# example:  python histogramSpecification.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\histogramSpecification.tif
#           python histogramSpecification.py --input E:/CVS/Project/Digital_Image_Processing/python/image/histogramSpecification.tif

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def histogramSpecification(imageInput):
    """
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    #------------------------------------------------------#
    # initially histogram specification
    G   = [0.0] * L # List of zeros - float
    pz  = [0.0] * L # List of zeros - float

    z1 = 0
    pz1 = 0.75
    z2 = 10
    pz2 = 7.0
    z3 = 20
    pz3 = 0.75
    z4 = 180
    pz4 = 0.0
    z5 = 200
    pz5 = 0.7
    z6 = 255
    pz6 = 0.0

    for z in range(0, L):
        if (z < z2):
            pz[z] = (pz2 - pz1) / (z2 - z1) * (z - z1) + pz1
        elif (z < z3):
            pz[z] = (pz3 - pz2) / (z3 - z2) * (z - z2) + pz2
        elif (z < z4):
            pz[z] = (pz4 - pz3) / (z4 - z3) * (z - z3) + pz3
        elif (z < z5):
            pz[z] = (pz5 - pz4) / (z5 - z4) * (z - z4) + pz4
        else:
            pz[z] = (pz6 - pz5) / (z6 - z5) * (z - z5) + pz5

    Sum = 0;
    for z in range(0, L):
        Sum += pz[z]
    for z in range(0, L):
        pz[z] = pz[z] / Sum
    
    for k in range(0, L):
        for i in range(0, k + 1):
            G[k] += pz[i]
        # G[k] *= (L - 1)

    #------------------------------------------------------#
    # histogram of input image
    pr  = [0.0] * L # List of zeros - float
    T   = [0.0] * L # List of zeros - float
    h   = [0]   * L # List of zeros - integer

    for x in range(0, M):
        for y in range(0, N):
            r = imageInput[x, y]
            h[r] += 1 # the number of pixels with intensity 'r'

    for r in range(0, L):
        pr[r] = h[r] / (M * N) # normalized histogram: the probability of occurrence of intensity level in a digital image

    for k in range(0, L):
        for j in range(0, k + 1):
            T[k] += pr[j]
        # T[k] *= (L - 1)

    # matching histogram
    for x in range (0, M):
        for y in range(0, N):
            r = imageInput[x, y]
            s = T[r];
            for k in range(0, L):
                if (G[k] >= s):
                    break
            # because the s[r] value is fractional,
            # so round it to its nearest integer value in the image
            imageOutput[x, y] = k

    return imageOutput

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,\
    help="input image path to processing")
args = vars(ap.parse_args())

# read original image from input image path
inputImagePath = args["input"]
imageFileName = ntpath.basename(inputImagePath)
imageOriginal = cv2.imread(inputImagePath)

# display original image 
cv2.imshow('original', imageOriginal)

# convert RGB to grayscale
imageGrayscale = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2GRAY)

# display grayscale image
cv2.imshow('grayscale', imageGrayscale)

# histogram specification processing
startTime = time.time()
imageHistogramSpecification = histogramSpecification(imageGrayscale)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display histogram specification processing image
cv2.imshow('imageHistogramSpecification', imageHistogramSpecification)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "histogramSpecification_" + imageFileName
    cv2.imwrite(resultPath, imageHistogramSpecification)