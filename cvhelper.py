__author__ = 'eclarson'

MEDIAN_PCT = 0.5
MEDIAN_DIF = 18

import cv2
import numpy as np
import itertools


def pre_process(img_input):
    img_input = cv2.pyrDown(img_input)  # convert down a bit for speedier processing
    img_input = cv2.flip(img_input, 1)  # mirror the image

    hsv_output = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_output)
    idx = np.argmax(v)

    v = normalize_array(v, idx)

    img_input = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    return img_input


def normalize_array(a, n_val):
    mx = a.item(n_val)
    a[np.where(a >= mx)] = mx
    a = 255*(a/float(a.item(n_val)))
    return a.astype(np.uint8)


def convert_color(img_input):
    hsv_output = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)  # convert to HSV
    return hsv_output


def set_img_boxes(img_in, corners, length):
    # set the boxes in side the image
    for col, row in corners:
        row = int(row)
        col = int(col)
        for x in itertools.product(range(row - length, row + length), range(col - length, col + length), range(0, 3)):
            img_in.itemset(x[0], x[1], x[2], 255)

    return img_in


def find_median_colors(img_in, corners, length):
    patches = []
    for col, row in corners:
        row = int(row)
        col = int(col)
        hue = []
        sat = []
        val = []
        for x in itertools.product(range(row - length, row + length), range(col - length, col + length)):
            hue.append(img_in.item(x[0], x[1], 0))
            sat.append(img_in.item(x[0], x[1], 1))
            val.append(img_in.item(x[0], x[1], 2))

        patches.append((np.median(hue), np.median(sat), np.median(val)))

    return patches


def segment_skin(img_in, median_values_in):
    img_in = cv2.GaussianBlur(img_in, (13, 13), 0)
    img_in = cv2.medianBlur(img_in, 5)
    width, height, depth = img_in.shape
    img_binary = np.zeros((width, height), np.uint8)
    for medians in median_values_in:
        lower = [int(np.max([med*(1-MEDIAN_PCT), med-MEDIAN_DIF])) for med in medians]
        upper = [int(np.min([med*(1+MEDIAN_PCT), med+MEDIAN_DIF])) for med in medians]
        lower[2] = 10
        upper[2] = 240

        img_seg = cv2.inRange(img_in, np.array(lower), np.array(upper))
        img_binary = cv2.bitwise_or(img_binary, img_seg)

    kernel = np.ones((9, 9), np.uint8)
    return cv2.morphologyEx(cv2.medianBlur(img_binary, 5), cv2.MORPH_CLOSE, kernel)


def find_contours(thresh_img):
    _, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    (h, w) = thresh_img.shape
    drawing = np.zeros((h, w, 3), np.uint8)
    max_area = 0

    if contours is not None and len(contours) > 0:
        ci = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            bb = cv2.boundingRect(cnt)
            ratio_shape = float(bb[2])/bb[3]

            if area > max_area and area > 10000 and (0.5 < ratio_shape < 1.5):
                max_area = area
                ci = i

        if max_area == 0:  # nothing to do
            return drawing

        cnt = contours[ci]
        hull = cv2.convexHull(cnt)

        cv2.drawContours(drawing, [cnt], 0, [0, 255, 0], 2)
        cv2.drawContours(drawing, [hull], 0, [128, 0, 0], 2)

        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)

        if True:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(drawing, start, end, [128, 128, 128], 2)
                    cv2.circle(drawing, far, 5, [0, 128, 128], -1)
                    if d > 5000:
                        cv2.putText(drawing, str(d), start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])

    return drawing
