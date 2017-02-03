ESC_KEY = 27  # for allowing the user to exit the program
C_KEY = ord('c')
U_KEY = ord('u')

import cv2
import cvhelper  # some helper files written by eclarson


def set_patches(img_in):
    height, width, depth = img_in.shape

    patches = [(width / 2, 3 * height / 4), (width / 2, height / 2), (width / 2, height / 4),
               (width / 2 + 45, 5 * height / 8), (width / 2 - 10, 5 * height / 8 + 10),
               (width / 2 + 25, 5 * height / 8 - 20), (width / 2 - 35, 5 * height / 8)]

    return patches


cap = cv2.VideoCapture(0)  # video capture on default input stream
if cap.isOpened():
    print "Video opened, getting first frame"
    ret, img = cap.read()  # get a frame from the video card

k = -1
calibrated = False
while cap.isOpened():
    ret, img = cap.read()  # get a frame from the video card
    if ret and (img is not None):
        img = cvhelper.pre_process(img)
        hsv_img = cvhelper.convert_color(img)
        hand_patches = set_patches(img)

        if k == C_KEY:
            # user initiated a calibrate command
            median_values = cvhelper.find_median_colors(hsv_img, hand_patches, 10)
            calibrated = True
            print median_values

        elif k == U_KEY:
            calibrated = False

        if not calibrated:
            # show the user where to place their hand
            img = cvhelper.set_img_boxes(img, hand_patches, 10)
        else:
            # process where we think the hand is
            skin_img = cvhelper.segment_skin(hsv_img, median_values)
            cv2.imshow('skin', skin_img)
            hsv_img = cvhelper.find_contours(skin_img)

        cv2.imshow('output', hsv_img)
        cv2.imshow('input', img)

    # get possible text input from the open figures
    # if a user presses a key, it will show up here
    k = cv2.waitKey(10)
    if k == ESC_KEY:
        break


