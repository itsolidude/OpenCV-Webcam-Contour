import cv2
import numpy as np
from matplotlib import pyplot as plt

#####################################
winWidth = 640
winHeight = 840
brightness = 100

cap = cv2.VideoCapture(0)
cap.set(3, winWidth)
cap.set(4, winHeight)
cap.set(10, brightness)

kernel = (7, 7)
faceCascade = cv2.CascadeClassifier("resources/cascades/haarcascade_frontalface_alt_tree.xml")
imgBlank = np.zeros((640, 840, 3), np.uint8)


#######################################################################
def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("cVal", "TrackBars", 10, 40, empty)
cv2.createTrackbar("bSize", "TrackBars", 77, 154, empty)


def preprocessing(frame, value_BSize, cVal):
    # imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(frame, kernel, 4)
    # gaussC = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, value_BSize,
    #                                cVal)
    # imgDial = cv2.dilate(gaussC, kernel, iterations=3)
    # imgErode = cv2.erode(imgDial, kernel, iterations=1)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    tv, thresh = cv2.threshold(lab[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh


def getFace():
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 2, 9)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)


def getContours(imPrePro):
    contours, hierarchy = cv2.findContours(imPrePro, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60:
            cv2.drawContours(imgCon, cnt, -1, (0, 255, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.07 * peri, True)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


######################################################################################################

while cap.isOpened():
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
        getFace()
        output = stackImages(0.7, ([imPrePro, imgCon, frame], [imgBlank, imgBlank, imgBlank]))
        cv2.imshow("Output", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

# lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
# tv, thresh = cv2.threshold(lab[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(thresh)
