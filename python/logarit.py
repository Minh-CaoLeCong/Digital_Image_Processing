# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python logarit.py --input imagePath
# example:  python logarit.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\logarit.tif
#           python logarit.py --input E:/CVS/Project/Digital_Image_Processing/python/image/negative.tif

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def logarit(imageInput):
    """
    c is a positive constant :
    if r = 0, then s = 0
	and if r = L - 1, then s = L - 1
	so from the form of the log transformation : s = c * log(1 + r)
	we can compute : c = (L - 1) / log(1 + L - 1)
	L = 256 = > c = 45.9859
    """
    # the possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    if (imageInput.shape[-1] == 3): # '-1' retrieves last item that is number of channels image, or could use '-2'
        imageInput = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY) #Convert RGB image to Gray image

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    c = float((L - 1) / math.log(L))

    # throughout each pixel of image
    for x in range(0, M):
        for y in range(0, N):
            # denote the values of pixels, before and after processing, by r and s, respectively
            r = imageInput[x, y]
            if (r==0):
                r = 1
            # log transformation function:
            s = float(c * math.log(1 + r))
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

# negative processing
startTime = time.time()
imageNegative = logarit(imageGrayscale)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display negative processing image
cv2.imshow('negative', imageNegative)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "logarit_" + imageFileName
    cv2.imwrite(resultPath, imageNegative)