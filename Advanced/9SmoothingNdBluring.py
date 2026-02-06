import cv2 as cv

img = cv.imread("Photos/cats.jpg")
cv.imshow("Cats",img)

# AVERAGE BLUR

# Kernel Size: It is basically a window that slides over the image
# The kernel size should be odd and positive, for example (3,3), (5,5), (7,7)
# The larger the kernel size, the more the image will be blurred

# In average blur, each pixel is replaced by the average of the pixels in the kernel window
# For example, if the kernel size is (3,3), each pixel will be replaced by the average of the 9 pixels in the kernel window
average = cv.blur(img, (5,5))
cv.imshow("Average Blur", average)


# GAUSSIAN BLUR (bletter than average blur) (but does less blurring than average blur)

# In Gaussian blur, each pixel in the kernel window is multiplied by a Gaussian function, which gives
# the pixels closer to the center a higher weight, while the pixels farther away a lower weight
# The center pixel is given the highest weight, and the weights decrease as you move away from the center
# This results in a smoother blur effect compared to average blur 

gaussian = cv.GaussianBlur(img, (5,5), 0)
# The third parameter is the standard deviation in the x and y directions, which controls the amount of blur
cv.imshow("Gaussian Blur", gaussian)



# MEDIAN BLUR (better than both average and gaussian blur)

# In median blur, each pixel is replaced by the median of the pixels in the kernel window
# This is useful for removing salt and pepper noise from the image
# salt and pepper noise is a type of noise that appears as random white and black pixels in the image

median = cv.medianBlur(img, 5)
cv.imshow("Median Blur", median)
# The kernel size should be odd and positive, for example 3, 5, 7, etc.
# The median filter does less blurring than average and gaussian blur, but is better at removing noise

# Median blur is not meant for high kernel sizes, like 7 or even 5 in some cases, as it can lead to loss of detail in the image
# It is best to use median blur for reducing some of the noise in the image


# BILATERAL BLUR (best for edge preservation)

bilateral = cv.bilateralFilter(img, 9, 75, 75)
# The first parameter is the source image, the second is the diameter of the pixel neighborhood,
# the third is the sigma value in the color space, which controls the amount of blur,
# and the fourth is the sigma value in the coordinate space, which means that 
# the pixels farther away from the center will have less influence on the final pixel value
cv.imshow("Bilateral Blur", bilateral)


cv.waitKey(20000)