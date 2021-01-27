import sys  # system functions (ie. exiting the program)
import os  # operating system functions (ie. path building on Windows vs. MacOs)
import time  # for time operations
import uuid  # for generating unique file names
import math  # math functions

from IPython.display import display as ipydisplay, Image, clear_output, HTML  # for interacting with the notebook better

import numpy as np  # matrix operations (ie. difference between two matricies)
import cv2  # (OpenCV) computer vision functions (ie. tracking)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))





###################################################################################################

ERODE = True

fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.createBackgroundSubtractorKNN()

video = cv2.VideoCapture(0)

while True:
    time.sleep(0.025)

    timer = cv2.getTickCount()

    # Read a new frame
    success, frame = video.read()
    if not success:
        # Frame not successfully read from video capture
        break

    fgmask = fgbg.apply(frame)

    # Apply erosion to clean up noise
    if ERODE:
        fgmask = cv2.erode(fgmask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(fgmask, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Display result
    cv2.imshow("fgmask", fgmask)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break  # ESC pressed

cv2.destroyAllWindows()
video.release()

