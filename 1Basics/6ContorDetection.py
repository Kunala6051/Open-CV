import cv2 as cv
import numpy as np

img = cv.imread("Photos/cats.jpg")
cv.imshow("Cats", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
cv.imshow("blur", blur)

canny = cv.Canny(img, 125, 175)
cv.imshow("canny", canny)

canny_blur = cv.Canny(blur, 125, 175)
cv.imshow("Canny_blur", canny_blur)


# CONTOUR: a contour is a curve joining all the continuous points (along a boundary)

contour, hierachy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# Retrieving all contours and storing them in the variable contour
# The second parameter is the contour retrieval mode, cv.RETR_ LIST retrieves all contours without establishing any hierarchical relationships
# cv.RETR_EXTERNAL retrieves only the external contours, cv.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours
# The third parameter is the contour approximation method, cv.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points

# hierachy is an array that contains information about the hierarchy of the contours
# It is useful for finding the parent-child relationship between contours

# contour is a list of all the contours found in the image
# Each contour is a numpy array of (x,y) coordinates of the points along the contour
print(f"{len(contour)} contours found in canny!")


contour1, hierachy = cv.findContours(canny_blur, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)    

print(f"{len(contour1)} contours found in canny_blur!")



# Binarizing the image (Black and White only)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# The first parameter is the source image, the second is the threshold value 
# which means that all pixel values below this value will be set to 0 (black) 
# and all pixel values above this value will be set to the maximum value (255 for white),
# the third is the maximum value to use with the THRESH_BINARY thresholding type, which is 255 in this case.

cv.imshow("Threshold", thresh)



# drawing contours
blank = np.zeros(img.shape, dtype='uint8')  # Create a blank image with the same shape as the original image
# The dtype is set to 'uint8' to match the original image type

cv.drawContours(blank, contour, -1, (0, 255, 0), thickness=1)
# The first parameter is the image on which to draw the contours, 
# the second is the contours which we want to draw (contour of canny image in this case),
# the third is the index of the contour to draw (-1 means draw all contours),
# the fourth is the color of the contours (green in this case),
# the fifth is the thickness of the contours (1 pixel in this case)
cv.imshow("Contours of Canny", blank)

blank2 = np.zeros(img.shape, dtype='uint8') 
cv.drawContours(blank2, contour1, -1, (255, 0, 255), thickness=1)

cv.imshow("Contours of Canny Blur", blank2)



cv.waitKey()