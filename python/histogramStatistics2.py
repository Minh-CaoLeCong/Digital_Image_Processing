# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python histogramStatistics2.py --input imagePath
# example:  python histogramStatistics2.py --input .\\image\\histogramStatistics.tif --c 22.8 --lo 2.0 --up 50.0
#           python histogramStatistics2.py --input ./image/histogramStatistics.tif

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def histogramStatistics2(imageInput, neighborhoodHeight, neighborhoodWidth, C, lower, upper):
    """
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    neighborhood = np.zeros((neighborhoodHeight, neighborhoodWidth), np.uint8)

    a = int(neighborhoodHeight / 2)
    b = int(neighborhoodWidth / 2)

    # C = 22.8

    for x in range(a, M - a):
        for y in range(b, N - b):
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    neighborhood[s + a, t + b] = imageInput[x + s, y + t]

            meanLocal = cv2.mean(neighborhood)

            # if ((k0 * meanGlobal <= meanLocal) and (meanLocal <= k1 * meanGlobal) and
            # ((k2 * varianceGlobal <= varianceLocal) and (varianceLocal <= k3 * varianceGlobal))):
            #     imageOutput[x, y] = int(C * imageInput[x, y])
            # else:
            #     imageOutput[x, y] = imageInput[x, y]

            if (lower <= meanLocal[0] and meanLocal[0] <= upper):
                imageOutput[x, y] = int(C * imageInput[x, y])
            else:
                imageOutput[x, y] = imageInput[x, y]

    return imageOutput

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,\
    help="input image path to processing")
ap.add_argument("-nh", "--height", type=int, default=3,\
    help="height of neighborhood - default: 3")
ap.add_argument("-nw", "--width", type=int, default=3,\
    help="width of neighborhood - default: 3")
ap.add_argument("-c", "--c", type=float, default=22.8,\
    help="specified constant, to increase (or decrease) the value of its intensity level\
        relative to the rest of the image - default: 22.8")
ap.add_argument("-lo", "--lo", type=float, default=2.0,\
    help="lower limit of the mean intensity - default: 2.0")
ap.add_argument("-up", "--up", type=float, default=50.0,\
    help="upper limit of the mean intensity - default: 50.0")
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

# local enhancement using histogram statistics processing
h = int(args["height"]) # height of neighborhood
w = int(args["width"]) # width of neighborhood
C = float(args["c"])
lo = float(args["lo"])
up = float(args["up"])
startTime = time.time()
imageHistogramStatistics2 = histogramStatistics2(imageGrayscale, h, w, C, lo, up)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display histogram statistics processing image
cv2.imshow('imageHistogramStatistics2', imageHistogramStatistics2)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "histogramStatistics2_" + imageFileName
    cv2.imwrite(resultPath, imageHistogramStatistics2)