import cv2 as cv


# For images, videos and live videos
def rescaleFrame(frame, scale=0.25): # default scale = 0.75
    new_width = int(frame.shape[1]* scale) # frame[0] is height and frame[1] is width
    new_height = int(frame.shape[0]* scale)

    dimensions = (new_width, new_height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)




# Create VideoCapture object for webcam (0 = default camera)
capture = cv.VideoCapture(0)

# Only for Live Videos
def ChangeResolution(width, height):
    capture.set(3,width) # 3 is for width and 4 is for height 
    # diff integers associated with different property for this method (eg. 10 for brightness)
    capture.set(4,height)

# rescaling photo
img = cv.imread("Photos/cat.jpg")

cv.imshow("Cat", img) 

resized_image = rescaleFrame(img, 0.75)
cv.imshow("Resized_Cat", resized_image)



print("started")
# rescaling video
vid = cv.VideoCapture("Videos/dog.mp4")
i=1
while True:
    isTrue, frame=vid.read() 

    cv.imshow('Video', frame)
    
    # print("Frame", i)
    # i=i+1

    newFrame = rescaleFrame(frame) 
    cv.imshow('Video Resized', newFrame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

vid.release()
cv.destroyAllWindows()



