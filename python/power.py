# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python power.py --input imagePath --gamma floatValue
# example:  python power.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\power.tif --gamma 0.3
#           python power.py --input E:/CVS/Project/Digital_Image_Processing/python/image/power.tif --gamma 0.3

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def power(imageInput, gamma):
    """
    NOTE: denote the values of pixels, before and after processing, by r and s, respectively
	'c' and 'gamma' is a positive constant.
	we can see:
		if 'gamma' < 1, make the image brighter
		if 'gamma' > 1, make the image darker

		if r = 0,		then s = 0
	and	if r = L - 1,	then s = L - 1
	so the form of Power-law transformations: s = c * pow(r, gamma)
	we can compute: c = (L - 1) / pow(L - 1, gamma) or c = pow(L - 1, 1 - gamma)
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    c = math.pow(L - 1, 1 - gamma)

    # throughout each pixel of image
    for x in range(0, M):
        for y in range(0, N):
            # denote the values of pixels, before and after processing, by r and s, respectively
            r = imageInput[x, y]
            if (r==0):
                r = 1
            # power-law transformation function:
            s = c * math.pow(r, gamma)
            imageOutput[x, y] = int(s)
    
    return imageOutput

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,\
    help="input image path to processing")
ap.add_argument("-g", "--gamma", required=True,\
    help="gamma constant")
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

# power-law processing
gamma = float(args["gamma"])
startTime = time.time()
imagePower = power(imageGrayscale, gamma)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display power processing image
cv2.imshow('power', imagePower)

print("[INFOR]: Press 's' key to save result and quit")
print("[INFOR]: Press 'q' key to quit and NOT save")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "power" + imageFileName
    cv2.imwrite(resultPath, imagePower)