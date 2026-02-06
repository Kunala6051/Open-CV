import cv2 as cv
import numpy as np

img = cv.imread("Photos/park.jpg")
cv.imshow("Park", img)

# Split the image into its color channels
b, g, r = cv.split(img)

# Show each channel
cv.imshow("Blue Channel", b) 
# The parts of the image that contain blue will have higher intensity values in the blue channel
# i.e. the darker parts of the image will have lower intensity values in the blue channel
# The same applies to the green and red channels
cv.imshow("Green Channel", g)
cv.imshow("Red Channel", r)

print(f"Image shape: {img.shape}")
# Image shape will be (height, width, channels)
# For example, if the image is 600x400 pixels, the shape will be (600, 400, 3)
# The last dimension is the number of color channels, which is 3 for a BGR image

# Each channel will have the same height and width as the original image, but only one color
# The shape of each channel will be (height, width)
# For example, if the image is 600x400 pixels, the shape of each channel will be (600, 400)
print(f"Blue channel shape: {b.shape}")
print(f"Green channel shape: {g.shape}")
print(f"Red channel shape: {r.shape}")

# Merge the channels back together
merged = cv.merge((b, g, r))  
cv.imshow("Merged Image", merged)


# Create a blank image with the same shape as the original image
blank = np.zeros(img.shape[:2], dtype='uint8') 
# The shape[:2] is used to get the height and width of the image, ignoring the color channels
# The dtype is set to 'uint8' to match the original image type
# This will create a blank image with the same height and width as the original image, but with only one channel (grayscale)

# Show the blank image
cv.imshow("Blank Image", blank)

# Create a 3-channel image with only the blue channel
blue_channel = cv.merge([b, blank, blank])
cv.imshow("Blue Channel Image", blue_channel)
# Again more the intensity of blue in the image, more the intensity of blue in the blue channel image


# Create a 3-channel image with only the green channel
green_channel = cv.merge([blank, g, blank])
cv.imshow("Green Channel Image", green_channel)

# Create a 3-channel image with only the red channel
red_channel = cv.merge([blank, blank, r])
cv.imshow("Red Channel Image", red_channel)

cv.waitKey(0)