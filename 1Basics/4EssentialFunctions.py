import cv2 as cv

img = cv.imread("Photos/park.jpg")
cv.imshow("Park", img)

# 1. Converting to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Park", gray)

# 2. Bluring
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# second parameter is the kernel size, it must be odd and positive
# higher the kernel size, more the blurring
cv.imshow("Blur Park", blur)


# 3. Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Edges", canny)
# u can also use cv.Canny(blur, 125, 175) to reduce noise before edge detection
CannyBlur = cv.Canny(blur, 125, 175)
cv.imshow("Canny Edges with Blur", CannyBlur)


# 4. Dilating the image
# Dilation increases the white region in the image or increases the size of the foreground object
# It is useful for closing small holes in the foreground
dilated = cv.dilate(CannyBlur, (7, 7), iterations=3)
# The second parameter is the kernel size, it must be odd and positive
# The third parameter is the number of iterations
cv.imshow("Dilated Edges", dilated)


# 5. Eroding the image
# Erosion reduces the white region in the image or reduces the size of the foreground object
# It is useful for removing small noise
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow("Eroded Edges", eroded)


# 6. Resize the image
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# The third parameter is the interpolation method
# cv.INTER_CUBIC is good for enlarging images, cv.INTER_AREA is good for shrinking images
# cv.INTER_LINEAR is good for both enlarging and shrinking images
cv.imshow("Resized Park", resized)


# 7. Cropping the image
cropped = img[50:200, 200:400]  # [y1:y2, x1:x2] as the image is a matrix
# This crops the image from y=50 to y=200 and x=200 to x=400
cv.imshow("Cropped Park", cropped)

cv.waitKey(0)