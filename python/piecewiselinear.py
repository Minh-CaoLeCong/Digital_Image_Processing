# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python piecewiselinear.py --input imagePath
# example:  python piecewiselinear.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\piecewiselinear.tif
#           python piecewiselinear.py --input E:/CVS/Project/Digital_Image_Processing/python/image/piecewiselinear.tif

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def piecewiselinear(imageInput):
    # the possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    # if not grayscale image, convert it
    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    # 'rmin' and 'rmax' denote the the minimum and maximum intensity
    # levels in the input image.
    # Using 'minMaxLoc' function in OpenCV to finds the minimum
    # and maximum element values in the input image
    rmin, rmax, _, _, = cv2.minMaxLoc(imageInput)

    # contrast stretching: 
	#   setting (r1, s1) = (rmin, 0) and (r2, s2) = (rmax, L - 1)
	# thresholding function:
	# 	setting (r1, s1) = (m, 0) and (r2, s2) = (m, L - 1)
	# 	where m is the mean intensity level in the image
    r1 = rmin; s1 = 0
    r2 = rmax; s2 = L - 1

    # throughout each pixel of image
    for x in range(0, M):
        for y in range(0, N):
            # denote the values of pixels, before and after processing, by r and s, respectively
            r = imageInput[x, y]
            # piecewise linear transformation function:
            if (r < r1):
                s = s1 / r1 * r
            elif (r < r2):
                s = (s2 - s1) / (r2 - r1) * (r - r1) + s1;
            else:
                s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
            imageOutput[x, y] = int(s)
    
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

# piecewiselinear processing
startTime = time.time()
imagePiecewiselinear = piecewiselinear(imageGrayscale)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display piecewiselinear processing image
cv2.imshow('piecewiselinear', imagePiecewiselinear)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "piecewiselinear_" + imageFileName
    cv2.imwrite(resultPath, imagePiecewiselinear)