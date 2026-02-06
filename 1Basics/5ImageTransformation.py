import cv2 as cv
import numpy as np

img = cv.imread("Photos/park.jpg")
cv.imshow("Park", img)

# 1. Translation
def translate(img, x, y):
    transformedMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])  # width, height
    return cv.warpAffine(img, transformedMat, dimensions)
    # wrapAffine applies an affine transformation to the image
    # The transformation matrix is a 2x3 matrix that defines the translation

# -y: up
# -x: left
# +y: down 
# +x: right

translated = translate(img, -100, 100)
cv.imshow("Translated Park", translated)




# 2. Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2] 
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)  # center of the image
    transformedMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0) # third parameter is the scale factor which is 1.0 (no scaling)
    dimensions = (width, height)
    return cv.warpAffine(img, transformedMat, dimensions)   

rotated = rotate(img, 45) # 45 for anticlockwise rotation and -45 for clockwise rotation
cv.imshow("Rotated Park", rotated)

rotated_rotated = rotate(rotated, 45)
cv.imshow("Rotated Rotated Park", rotated_rotated)




# 3. Resize the image
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# The third parameter is the interpolation method
# cv.INTER_CUBIC is good for enlarging images, cv.INTER_AREA is good for shrinking images
# cv.INTER_LINEAR is good for both enlarging and shrinking images
cv.imshow("Resized Park", resized)



# 4. Flipping the image
flip = cv.flip(img, 1)  # 0 for vertical flip, 1 for horizontal flip, -1 for both
cv.imshow("Flipped Park", flip)



# 5. Cropping the image
cropped = img[50:200, 200:400]  # [y1:y2, x1:x2] as the image is a matrix
# This crops the image from y=50 to y=200 and x=200 to x=400
cv.imshow("Cropped Park", cropped)


cv.waitKey(0)