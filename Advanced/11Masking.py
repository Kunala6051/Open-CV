import cv2 as cv
import numpy as np

# Masking is a technique used to isolate a specific region of an image for further processing
img = cv.imread("Photos/cats.jpg")
cv.imshow("Cats", img)

# Create a blank mask with the same dimensions as the image
circle_mask1 = np.zeros(img.shape[:2], dtype='uint8') # The shape[:2] is used to get the height and width of the image, ignoring the color channels

# Define a circular region of interest (ROI) in the mask
cv.circle(circle_mask1, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# The circle is drawn at the center of the image with a radius of 100 pixels
# shape[1] is the width and shape[0] is the height of the image

masked_img = cv.bitwise_and(img, img, mask=circle_mask1)
# for each pixel in the image, if the corresponding pixel in the mask is white (255), the pixel is kept; otherwise, it is set to black (0)

cv.imshow("Masked Image", masked_img)




blank = np.zeros(img.shape[:2], dtype='uint8')
circle_mask2 = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
rectangle_mask = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

weired_shape = cv.bitwise_and(circle_mask2, rectangle_mask)
# This will create a mask that is the intersection of the circle and rectangle masks

cv.imshow("Circle Mask", circle_mask2)
cv.imshow("Rectangle Mask", rectangle_mask)
cv.imshow("Weird Shape", weired_shape)

masked_img2 = cv.bitwise_and(img, img, mask=weired_shape)
# This will apply the weird shape mask to the original image, keeping only the pixels that fall

cv.imshow("Masked Image 2", masked_img2)


# JUST REMEMBER: THE SIZE OF THE MASK SHOULD BE THE SAME AS THE SIZE OF THE IMAGE
# If the mask is smaller or larger than the image, it will not work correctly

cv.waitKey(0)