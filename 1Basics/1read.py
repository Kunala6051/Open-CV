import cv2 as cv

img = cv.imread("Photos/cat.jpg") # returns a matrix of pixels of the image
# this is an image of (640 x 427) dimensions

cv.imshow("Cat", img) # window name=Cat on which image will be shown and the name of the matrix of pixels
# cv.waitKey(0)


# Lets try an enlarged image of (2400 x 1600)
enlarged_img = cv.imread("Photos/cat_large.jpg")
cv.imshow("Enlarged_Cat", enlarged_img)
cv.waitKey(0)
# It will go out of the screen 



# reading videos
vid = cv.VideoCapture("Videos/dog.mp4")
# here it is taking path as a paramter but,
# if your webcam is linked then it takes 0 (directing to webcam)
# if u want to use external cameras connected then it uses integers 1, 2, 3 _ _ accordingly for the cam u want to use

while True:
    isTrue, frame=vid.read() 
    # vid.read() reads the next frame from the video or webcam.
    # isTrue is True if a frame is read successfully, otherwise False.
    # frame contains the image (video frame).
    # while True: creates an infinite loop to continuously read and display frames.

    cv.imshow('Video', frame)
    #Displays the current frame in a window titled 'Video'.

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    # cv.waitKey(20) waits 20 milliseconds for a key press.
    # & 0xFF ensures compatibility across platforms.
    # ord('d') is the ASCII value of 'd'.
    # So: if you press the 'd' key, the loop will break (exit).

vid.release()
cv.destroyAllWindows()
# vid.release() releases the video or webcam resource.
# cv.destroyAllWindows() closes any OpenCV windows opened by imshow.


# Error: Assertion failed means the path u gave for the video did not find