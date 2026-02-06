import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread("Photos/cats.jpg")
cv.imshow("Cats", img)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# We have already done edge detection with canny 
canny = cv.Canny(gray, 125, 175)
cv.imshow("Canny", canny)

# Now we will use another methods to detect edges

# 1. Laplacian Edge Detection (Look like drawn by chalk)

lap = cv.Laplacian(gray, cv.CV_64F)
# The first parameter is the source image, the second is the depth of the output image
# cv.CV_64F is used to avoid overflow and underflow issues
# cv.CV_64F means that the output image will have a depth of 64 bits (double precision)

lap = np.uint8(np.absolute(lap))
# We need to convert the output to uint8 because the Laplacian function returns a float image
# np.absolute() is used to get the absolute value(+ve) of the Laplacian image

cv.imshow('Laplacian', lap)



# 2. Sobel Edge Detection (Looks like a pencil sketch)
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)  # Sobel in x direction
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)  # Sobel in y direction
# The first parameter is the source image, the second is the depth of the output image,
# cv.CV_64F is used to avoid overflow and underflow issues
# cv.CV_64F means that the output image will have a depth of 64 bits (double precision)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)

sobel_combined = cv.bitwise_or(sobelx, sobely)
# Combining the two Sobel images using bitwise OR operation

cv.imshow('Sobel Combined', sobel_combined)


# Canny is an advanced edge detection method that uses a multi-stage algorithm to detect edges in an image
# It even uses sobel as one of its steps to calculate the gradient of the image

cv.waitKey(0)