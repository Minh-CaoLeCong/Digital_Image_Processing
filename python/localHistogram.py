# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python localHistogram.py --input imagePath
# example:  python localHistogram.py --input .\\image\\localHistogram.tif
#           python localHistogram.py --input ./image/localHistogram.tif

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def histogramEqualization(imageInput):
    """
    the "normalized histogram" is defined as: 
	(the probability of occurrence of intensity level 'rk' in a image) where rk = [0, L - 1]
		p(rk) = h(rk) / M * N = nk / M * N
	NOTE: the sum of p(rk) for all values of k is always 1
	h(rk) = nk: 
	    where k = 0, 1, 2, ..., L - 1, denote the intensities of an L-level digital image.
	    'nk' is the number of pixels that have intensity 'rk'.
	    M and N are the number of image rows and columns.
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    # if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
    #     imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    h = [0] * L # List of zeros - integer

    for x in range(0, M):
        for y in range(0, N):
            r = int(imageInput[x, y])
            h[r] += 1 # the number of pixels with intensity 'r'

    p = [0.0] * L # List of zeros - float

    for r in range(0, L):
        p[r] = h[r] / (M * N) # normalized histogram: the probability of occurrence of intensity level in a digital image

    s = [0.0] * L # List of zeros - float

    for k in range(0, L):
        for j in range(0, k):
            s[k] += p[j]
        s[k] *= (L - 1)

    for x in range (0, M):
        for y in range(0, N):
            r = imageInput[x, y]
            # because the s[r] value is fractional,
            # so round it to its nearest integer value in the image
            imageOutput[x, y] = int(s[r])

    return imageOutput

def localHistogram(imageInput):
    """
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    neighborhoodHeight = 3
    neighborhoodWidth = 3

    neighborhoodInput = np.zeros((neighborhoodHeight, neighborhoodWidth), np.uint8)
    neighborhoodOutput = np.zeros((neighborhoodHeight, neighborhoodWidth), np.uint8)

    a = int(neighborhoodHeight / 2)
    b = int(neighborhoodWidth / 2)

    for x in range(a, M - a):
        for y in range(b, N - b):
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    neighborhoodInput[s + a, t + b] = imageInput[x + s, y + t]
            # neighborhoodOutput = histogramEqualization(neighborhoodInput) # take lots of time to excecute than cv2.equalizeHist
            neighborhoodOutput = cv2.equalizeHist(neighborhoodInput)
            imageOutput[x, y] = neighborhoodOutput[a, b]

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

# local histogram processing
startTime = time.time()
imageLocalHistogram = localHistogram(imageGrayscale)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display local histogram processing image
cv2.imshow('imageLocalHistogram', imageLocalHistogram)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "localHistogram_" + imageFileName
    cv2.imwrite(resultPath, imageLocalHistogram)