# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python greyWorld.py --input imagePath
# example:  python greyWorld.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\greyWorld.jpg
#           python greyWorld.py --input E:/CVS/Project/Digital_Image_Processing/python/image/greyWorld.jpg

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def greyWorld(imageInput):
    """
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    imageOutput = np.zeros((imageInput.shape[0], imageInput.shape[1]), np.uint8)

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    imageInput_blueChannel, imageInput_greenChannel, imageInput_redChannel =\
        cv2.split(imageInput)
    
    mean_imageInput_blueChannel = cv2.mean(imageInput_blueChannel)
    mean_imageInput_greenChannel = cv2.mean(imageInput_greenChannel)
    mean_imageInput_redChannel = cv2.mean(imageInput_redChannel)

    scale = (mean_imageInput_blueChannel[0] + mean_imageInput_greenChannel[0] +\
                mean_imageInput_redChannel[0]) / 3

    imageInput_blueChannel = cv2.addWeighted(src1=imageInput_blueChannel,\
        alpha=(scale / mean_imageInput_blueChannel[0]), src2=0, beta=0, gamma=0)
    imageInput_greenChannel = cv2.addWeighted(src1=imageInput_greenChannel,\
        alpha=(scale / mean_imageInput_greenChannel[0]), src2=0, beta=0, gamma=0)
    imageInput_redChannel = cv2.addWeighted(src1=imageInput_redChannel,\
        alpha=(scale / mean_imageInput_redChannel[0]), src2=0, beta=0, gamma=0)

    imageOutput = cv2.merge([imageInput_blueChannel, imageInput_greenChannel,\
        imageInput_redChannel])

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

# color balance processing
startTime = time.time()
imageGreyWorld = greyWorld(imageOriginal)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display color balance processing image
cv2.imshow('Color Balance', imageGreyWorld)

print("[INFOR]: Press 's' key to save result and quit")
print("[INFOR]: Press 'q' key to quit and NOT save")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "greyWorld_" + imageFileName
    cv2.imwrite(resultPath, imageGreyWorld)