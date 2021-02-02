import cv2
import numpy as np

winWidth = 640
winHeight = 840
brightness = 100

cap = cv2.VideoCapture(0)
cap.set(3, winWidth)
cap.set(4, winHeight)
cap.set(10, brightness)

kernel = (7, 7)





while cap.isOpened():
