import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# Since, it is a binary image, so we dont have to specify the full color channel (BGR) just one color representing intensity of white (max 255) or black (0)
cir = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow("rectangle", rectangle)
cv.imshow("Circle", cir)


# Bitwise AND operation
bitwise_and = cv.bitwise_and(rectangle, cir)
# It will return a new image where the pixels are set to 255 (white) only where both images have white pixels
# In this case, it will return the intersection of the rectangle and circle
cv.imshow("Bitwise AND", bitwise_and)


# Bitwise OR operation
bitwise_or = cv.bitwise_or(rectangle, cir)
# It will return a new image where the pixels are set to 255 (white) where either of the images has white pixels
# In this case, it will return the union of the rectangle and circle
cv.imshow("Bitwise OR", bitwise_or)


# Bitwise XOR operation
bitwise_xor = cv.bitwise_xor(rectangle, cir)
# It will return a new image where the pixels are set to 255 (white) where either of the images has white pixels, but not both
# In this case, it will return the pixels that are in either the rectangle or the circle, but not in both
cv.imshow("Bitwise XOR", bitwise_xor)

# Bitwise NOT operation
bitwise_not_rectangle = cv.bitwise_not(rectangle)
bitwise_not_circle = cv.bitwise_not(cir)
# It will return a new image where the pixels are inverted, i.e. white pixels become black and black pixels become white
cv.imshow("Bitwise NOT Rectangle", bitwise_not_rectangle)
cv.imshow("Bitwise NOT Circle", bitwise_not_circle)



cv.waitKey(0)