# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python spatialCorrelation.py --input imagePath
# example:  python spatialCorrelation.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\spatialCorrelation.tif
#           python spatialCorrelation.py --input E:/CVS/Project/Digital_Image_Processing/python/image/spatialCorrelation.tif

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def spatialCorrelation(imageInput, kernel):
    """
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    m = kernel.shape[0] # height of image
    n = kernel.shape[1] # width of image

    a = int(m / 2)
    b = int(n / 2)

    for x in range(a, M - a):
        for y in range(b, N - b):
            r = 0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    r += kernel[s + a, t + b] * imageInput[x + s, y + t]
            imageOutput[x, y] = int(r)

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

# spatial correlation processing
startTime = time.time()
w = np.ones((3, 3), np.uint8)
print(w)
imageSpatialCorrelation = spatialCorrelation(imageGrayscale, w)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display spatial correlation processing image
cv2.imshow('spatialCorrelation', imageSpatialCorrelation)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "spatialCorrelation_" + imageFileName
    cv2.imwrite(resultPath, imageSpatialCorrelation)