# Tutor: Tran Tien Duc
# Gmail: trantienduc@gmail.com
# Created by Cao Le Cong Minh
# Gmail: caolecongminh1997@gmail.com
# Github: https://github.com/Minh-CaoLeCong

# USAGE
# python histogramRGB.py --input imagePath
# example:  python histogramRGB.py --input E:\\CVS\\Project\\Digital_Image_Processing\\python\\image\\histogramRGB.jpg
#           python histogramRGB.py --input E:/CVS/Project/Digital_Image_Processing/python/image/histogramRGB.jpg

# import the necessary packages
import argparse
import cv2
import numpy as np
import ntpath
import time
import math

def histogramRGB(imageInput):
    """
    """
    # the number of possible intensity levels in the image (256 for an 8-bit image)
    L = 256

    M = imageInput.shape[0] # height of image
    N = imageInput.shape[1] # width of image

    imageInput_blueChannel, imageInput_greenChannel, imageInput_redChannel =\
        cv2.split(imageInput)

    blueChannel_hist = cv2.calcHist(imageInput_blueChannel, [0], None, [L], [0, L])
    greenChannel_hist = cv2.calcHist(imageInput_greenChannel, [0], None, [L], [0, L])
    redChannel_hist = cv2.calcHist(imageInput_redChannel, [0], None, [L], [0, L])

    hist_width = 512
    hist_height = 400
    bin_width = int(round(hist_width/L))

    imageOutputRGB = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    imageOutputRed = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    imageOutputGreen = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    imageOutputBlue = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    cv2.normalize(blueChannel_hist, blueChannel_hist, alpha=0, beta=hist_height, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(greenChannel_hist, greenChannel_hist, alpha=0, beta=hist_height, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(redChannel_hist, redChannel_hist, alpha=0, beta=hist_height, norm_type=cv2.NORM_MINMAX)

    for i in range(1, L):
        cv2.line(imageOutputRGB, (bin_width*(i-1), hist_height - int(np.round(blueChannel_hist[i-1]))),\
            ( bin_width*(i), hist_height - int(np.round(blueChannel_hist[i]))),\
                ( 255, 0, 0), thickness=2)
        cv2.line(imageOutputRGB, (bin_width*(i-1), hist_height - int(np.round(greenChannel_hist[i-1]))),\
            ( bin_width*(i), hist_height - int(np.round(greenChannel_hist[i]))),\
                ( 0, 255, 0), thickness=2)
        cv2.line(imageOutputRGB, ( bin_width*(i-1), hist_height - int(np.round(redChannel_hist[i-1]))),\
            ( bin_width*(i), hist_height - int(np.round(redChannel_hist[i]))),\
                ( 0, 0, 255), thickness=2)
        cv2.line(imageOutputBlue, (bin_width*(i-1), hist_height - int(np.round(blueChannel_hist[i-1]))),\
            ( bin_width*(i), hist_height - int(np.round(blueChannel_hist[i]))),\
                ( 255, 0, 0), thickness=2)
        cv2.line(imageOutputGreen, (bin_width*(i-1), hist_height - int(np.round(greenChannel_hist[i-1]))),\
            ( bin_width*(i), hist_height - int(np.round(greenChannel_hist[i]))),\
                ( 0, 255, 0), thickness=2)
        cv2.line(imageOutputRed, ( bin_width*(i-1), hist_height - int(np.round(redChannel_hist[i-1]))),\
            ( bin_width*(i), hist_height - int(np.round(redChannel_hist[i]))),\
                ( 0, 0, 255), thickness=2)

    return imageOutputRGB, imageOutputRed, imageOutputGreen, imageOutputBlue

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

# histogramRGB processing
startTime = time.time()
imageHistogramRGB, imageHistogramRed, imageHistogramGreen, imageHistogramBlue = histogramRGB(imageOriginal)
print("[INFOR]: Time execution: {}".format(time.time() - startTime))

# display histogramRGB processing image
cv2.imshow('histogramRGB', imageHistogramRGB)
cv2.imshow('histogramRed', imageHistogramRed)
cv2.imshow('histogramGreen', imageHistogramGreen)
cv2.imshow('histogramBlue', imageHistogramBlue)

print("[INFOR]: Press 's' key to save result")
print("[INFOR]: Press 'q' key to quit")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    resultPath = "./output/" + "histogramRGB_" + imageFileName
    cv2.imwrite(resultPath, imageHistogramRGB)