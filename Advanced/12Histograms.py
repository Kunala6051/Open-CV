import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Histogram is a graphical representation of the distribution of pixel intensities in an image
img = cv.imread("Photos/cats 2.jpg")
cv.imshow("Cats", img)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Calculate the histogram of the grayscale image
hist = cv.calcHist([gray], [0], None, [256], [0, 255])
# first argument is the image, second is the channel (0 for grayscale), third is the mask (None means no mask), 
# fourth is the number of bins,
# and fifth is the range of pixel values

# BINS EXPLAINATION: (4th parameter)

# histSize = [256] (imagine them like 256 boxes lined up next to each other)
# Then we have 256 bins — one for each possible pixel value (0, 1, 2, ..., 255), So

# Bin 0 → counts pixels with value 0
# Bin 1 → counts pixels with value 1
# ...
# Bin 255 → counts pixels with value 255

# Plot the histogram 
plt.figure() # Create a new figure for the plot
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Values")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist)
plt.show()

# In this histogram, the x-axis represents the pixel values (0-255) and the y-axis represents the frequency of each pixel value in the image
# In this image, a large number of pixels have a value around 50-60, indicating that there are many dark areas in the image


# CALCULATING HISTOGRAM FOR A MASKED REGION

blank = np.zeros((img.shape[:2]), dtype='uint8')

circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

mask = cv.bitwise_and(gray,gray,mask=circle)
cv.imshow("mask", mask)

hist1 = cv.calcHist([gray], [0], circle, [256], [0, 255]) # using the circle mask to calculate the histogram of the masked region

# Plot the histogram 
plt.figure() # Create a new figure for the plot
plt.title("Masked_img Histogram")
plt.xlabel("Pixel Values")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.plot(hist1)
plt.show()


# coloured histograms

plt.figure() # Create a new figure for the plot
plt.title("Coloured Histogram")
plt.xlabel("Pixel Values")
plt.ylabel("Frequency")


colors = ('b','g','r')

for i,col in enumerate(colors):   # enumerate(colors) gives us both the index and the color
    # cv.calcHist takes a list of images, so we pass [img] and [i] specifies the channel (0 for blue, 1 for green, 2 for red)
    # None means no mask, [256] is the number of bins, and [0, 256] is the range of pixel values
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()    

# coloured histograms for a mask

plt.figure() # Create a new figure for the plot
plt.title("Masked Histogram")
plt.xlabel("Pixel Values")
plt.ylabel("Frequency")



for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], circle, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()  

cv.waitKey(0)


