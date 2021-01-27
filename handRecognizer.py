import cv2
import numpy as np

#####################################
winWidth = 640
winHeight = 840
brightness = 100

cap = cv2.VideoCapture(0)
cap.set(3, winWidth)
cap.set(4, winHeight)
cap.set(10, brightness)

kernel = (5, 5)



#######################################################################
def empty(a):
    pass



cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("cVal", "TrackBars", 0, 20, empty)
cv2.createTrackbar("bSize", "TrackBars", 0, 20, empty)


def preprocessing(frame, value_BSize, cVal):
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(imgHsv, lower, upper)
    imgBlurred = cv2.GaussianBlur(imgGray, kernel, 3)
    gaussC = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, value_BSize, cVal)
    imgDial = cv2.dilate(gaussC, kernel, iterations=3)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)

    return imgDial


def getContours(imPrePro):
    contours, hierarchy = cv2.findContours(imPrePro, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60:
            cv2.drawContours(imgCon, cnt, -1, (255, 0, 0), 3)


#######################################################################################################

while (cap.isOpened()):
    success, frame = cap.read()
    cVal = cv2.getTrackbarPos("cVal", "TrackBars")
    bVal = cv2.getTrackbarPos("bVal", "TrackBars")
    value_BSize = cv2.getTrackbarPos("bSize", "TrackBars")
    value_BSize = max(3, value_BSize)
    if (value_BSize % 2 == 0):
        value_BSize += 1

    if success == True:
        frame = cv2.flip(frame, 1)
        imgCon = frame.copy()
        imPrePro = preprocessing(frame, value_BSize, cVal)
        getContours(imPrePro)
        cv2.imshow("Preprocessed", imPrePro)
        cv2.imshow("Original", imgCon)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
