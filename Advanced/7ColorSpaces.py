import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("Photos/park.jpg")
cv.imshow("Park", img)

# BGR to Gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# BGR to HSV
# HSV stands for Hue, Saturation, Value
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)

# BGR to L*a*b
# L*a*b stands for Lightness, a* (green to red), b* (blue to yellow)
# It is a color space that is designed to be perceptually uniform, meaning that the differences between colors are more consistent with human perception
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
cv.imshow("Lab", lab)

# You can also convert hsv to bgr, lab to bgr, and gray to bgr

# hsv To BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow("HSV to BGR", hsv_bgr)


# There is no function to convert hsv to lab or gray, you need to convert to BGR first


# Rememmber openCV uses BGR format by default, not RGB unlike most other libraries
# for eg: Matplotlib uses RGB format
# and if we want to display the image using Matplotlib, it will imagine it as RGB so we need to convert it to RGB first

plt.imshow(img)
plt.title("Original Image")
plt.axis("off")  # Hide the axes
plt.show()


# Now turning image to RGB for Matplotlib
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Image in RGB")
plt.axis("off")  # Hide the axes
plt.show()


# if we see the rgb image in opencv, it will look weird because opencv uses BGR format
cv.imshow("RGB Image", img_rgb)


cv.waitKey()