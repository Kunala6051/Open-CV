import cv2 as cv
import numpy as np

# Create a blank image with all zeros (black)
blank = np.zeros((500,500,3), dtype='uint8') 
# 500x500 pixels, 3 channels (BGR)
# The first two dimensions are the height and width, and the third dimension is for color channels
# dtype='uint8' indicates that the the array is of image type

cv.imshow('Blank', blank)

# Colouring

# full image
blank[:] = 0,255,0  # Fill the image with a green color
# Turning all pixels to green
cv.imshow('Green Blank', blank) 

# # partial image
# blank[200:300, 100:400] = 0,0,255  # Fill a rectangle with red color
# cv.imshow('Red Rectangle', blank)



# Drawing rectangle

cv.rectangle(blank, (0,0), (250,250), (255,0,0), thickness=2)  # Draw an outline of a blue rectangle
# The first argument is the image, the second is the top-left corner(origin), the third is the bottom-right corner,
# the fourth is the color in BGR format, and the fifth is the thickness of the rectangle
# If thickness is -1, it will fill the rectangle with the color
# or you can use cv.FILLED for filling

# u can also write it as cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (255,0,0), thickness=2)
# AS the shape of the image is (500,500,3), blank.shape[1] is 500 and blank.shape[0] is 500
# So, (blank.shape[1]//2, blank.shape[0]//2) is (250, 250)

cv.imshow('Blue Rectangle', blank)


# Drawing circle
cv.circle(blank, (250,250), 40, (255,255,255), thickness=-1)  # Draw an outline of a green circle
# The first argument is the image, the second is the center of the circle, the third is the radius,
# the fourth is the color in BGR format, and the fifth is the thickness of the circle
cv.imshow('Green Circle', blank)


# Drawing line
cv.line(blank, (0,250), (250,250), (0,0,0), thickness=3)  # Draw a black line from top-left to bottom-right
# The first argument is the image, the second is the starting point of the line, the third is the ending point,
# the fourth is the color in BGR format, and the fifth is the thickness of the line
cv.imshow('Black Line', blank)


# Writing text
cv.putText(blank, 'Hello, OpenCV!', (100, 250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=2)
# The first argument is the image, the second is the text to be written, the third is the position of the text,
# the fourth is the font type, the fifth is the font scale, the sixth is the color in BGR format, and the seventh is the thickness of the text
cv.imshow('Text', blank)

cv.waitKey(0)