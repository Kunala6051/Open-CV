import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



# Thresholding is a technique used to create binary images from grayscale images by setting a threshold value.
# Pixels with intensity values above the threshold are set to one value (usually white), and those below are set to another (usually black).
# This is useful for segmenting objects in an image, making it easier to analyze or process.

# There are several types of thresholding methods in OpenCV, we are going to study:
# 1. Simple Thresholding    
# 2. Adaptive Thresholding

img = cv.imread("Photos/cats.jpg")
cv.imshow("Cats", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# 1. Simple Thresholding

threshold, thresh1 = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# threshold is the value we are using to separate the pixel values into two groups
# thresh1 is the output binary image
# cv.THRESH_BINARY means that pixel values above the threshold(150) will be set to 255 (white) and those below will be set to 0 (black)
cv.imshow("Simple Threshold", thresh1)

# You can also use cv.THRESH_BINARY_INV to invert the binary image
threshold, thresh2 = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# This means that pixel values above the threshold will be set to 0 (black) and those below will be set to 255 (white)
cv.imshow("Simple Threshold Inverted", thresh2)


# 2. Adaptive Thresholding

# We dont have to specify a fixed threshold value, instead, the threshold value is calculated for smaller regions of the image.
# Adaptive thresholding means that the threshold value is calculated for smaller regions of the image, allowing for different threshold values in different areas.
# Here the kernel size is the size of the region we are considering to calculate the threshold value.
# It can be a mean value or the guassian mean value (we discussed it earlier)

adaptive_thresh_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# 255 is the maximum value to use with the THRESH_BINARY thresholding type
# cv.ADAPTIVE_THRESH_MEAN_C means that the threshold value is the mean of the pixel values in the neighborhood of the pixel
# 11 is the size of the neighborhood (it must be an odd number), and 2 is the constant subtracted from the mean
cv.imshow("Adaptive Threshold Mean", adaptive_thresh_mean)

adaptive_thresh_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# cv.ADAPTIVE_THRESH_GAUSSIAN_C means that the threshold value is the weighted sum of the pixel values in the neighborhood of the pixel
# The weights are calculated using a Gaussian function
cv.imshow("Adaptive Threshold Gaussian", adaptive_thresh_gaussian)


# Inverting the adaptive thresholding images
adaptive_thresh_mean_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
cv.imshow("Adaptive Threshold Mean Inverted", adaptive_thresh_mean_inv)

adaptive_thresh_gaussian_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
cv.imshow("Adaptive Threshold Gaussian Inverted", adaptive_thresh_gaussian_inv)


cv.waitKey(0)