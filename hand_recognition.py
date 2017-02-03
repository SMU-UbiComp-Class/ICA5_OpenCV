''' Adapted from online tutorial:
http://creat-tabu.blogspot.ro/2013/08/opencv-python-hand-gesture-recognition.html
'''

ESC_KEY = 27  # for allowing the user to exit the program

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # video capture on default input stream
if cap.isOpened():
    print "Video opened, getting first frame"
    ret, img = cap.read()  # get a frame from the video card

while cap.isOpened():
    ret, img = cap.read()  # get a frame from the video card
    if ret and (img is not None):
        img = cv2.pyrDown(img)  # convert down a bit for speedier processing
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  # blur the image a bit

        """ Now get a "binary image" and choose a gray value for which everything above the
        gray value will be "white" and everything below the gray value will be black.
        The hope is that we choose a "good" value such that the hand is always highlighted
        """
        ret, thresh_img = cv2.threshold(blur_img, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        drawing = np.zeros(img.shape, np.uint8)
        max_area = 0

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if area > max_area:
                max_area = area
                ci = i

        cnt = contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)

        if moments['m00'] != 0:
            cx = \
                int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

        center = (cx, cy)
        cv2.circle
        (img, center, 5, [0, 0, 255], 2)
        cv2.drawContours(drawing, [cnt], 0, ( 0, 255, 0 ), 2)
        cv2.drawContours(drawing, [hull], 0, ( 0, 0, 255 ), 2)

        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)

        if True:
            defects = cv2.convexityDefects(cnt, hull)
            mind = 0
            maxd = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                dist = cv2.pointPolygonTest(cnt, center, True)
                cv2.line(img, start, end, [0, 255, 0], 2)

                cv2.circle(img, far, 5, [0, 0, 255], -1)
                print i
                i = 0

        cv2.imshow('output', drawing)
        cv2.imshow('input', img)

    # get possible text input from the open figures
    # if a user presses a key, it will show up here
    k = cv2.waitKey(10)
    if k == ESC_KEY:
        break


