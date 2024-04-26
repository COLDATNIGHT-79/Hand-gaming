import cv2  
import imutils 
import numpy as np  
import time  
from control import Control as c  
from sklearn.metrics import pairwise  

cap = cv2.VideoCapture(1)

def nothing(x):
    pass

cv2.namedWindow('Result')

h, s, v = 0, 0,0,

cv2.createTrackbar('h', 'Result', 0, 255, nothing)
cv2.createTrackbar('s', 'Result', 0, 255, nothing)
cv2.createTrackbar('v', 'Result', 0, 255, nothing)
cv2.createTrackbar('Start', 'Result', 0, 1, nothing)

top, right, bottom, left = 0, 700, 525, 240

ob = c()

while 1:
    try:
        _, frame = cap.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        roi = frame[top:bottom, left:right]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        blurred = cv2.GaussianBlur(roi, (15, 15), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h = cv2.getTrackbarPos('h', 'Result')
        s = cv2.getTrackbarPos('s', 'Result')
        v = cv2.getTrackbarPos('v', 'Result')

        lower_skin = np.array([h, s, v])
        upper_skin = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        Result = cv2.bitwise_and(roi, roi, mask=mask)

        cv2.imshow('Result', Result)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(approx)
        cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)

        extreme_top = tuple(hull[hull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(hull[hull[:, :, 1].argmax()][0])
        extreme_left = tuple(hull[hull[:, :, 0].argmin()][0])
        extreme_right = tuple(hull[hull[:, :, 0].argmax()][0])

        cX = int((extreme_left[0] + extreme_right[0]) / 2)
        cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

        cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)
        cv2.circle(roi, (cX, cY), radius=5, color=(255, 0, 0), thickness=3)
        cv2.circle(roi, extreme_left, radius=3, color=(0, 0, 255), thickness=2)
        cv2.circle(roi, extreme_right, radius=3, color=(0, 0, 255), thickness=2)

        distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right])[0]
        max_distance = distances[distances.argmax()]

        x1, y1 = extreme_left
        x2, y2 = extreme_right
        slope = float((y2 - y1) / (x2 - x1))

        radius = int(0.8 * max_distance)

        circumference = (2 * np.pi * radius)

        millis = int(round(time.time() * 1000))
        seconds = millis / 5
        if seconds % 8 == 0:
            print("DISTANCE : " + str(round(max_distance, 2)))
            print("SLOPE : ", slope)

        start = cv2.getTrackbarPos('Start', 'Result')

        if start > 0:
            ob.startControlling(max_distance, slope)
        cv2.imshow("Final", frame)
    except:
        pass
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()