import cv2
import numpy as np
import time
import itertools

section = 0

# setup some windows for viewing
cv2.namedWindow("demowin1")
cv2.namedWindow("demowin2")

# open the video card for capture
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    print "vc opened, getting first frame"
    rval, frame = vc.read()
    # this will likely fail the first time
    # the webcam often needs some time to open fully
    key = 0
else:
    print "vc not open, exiting"
    key = 27

while key != 27 and vc.isOpened():  # the escape key and the capture device is open
    rval, frame = vc.read()
    key = cv2.waitKey(10)

    # interpret the input key, on top number line of keyboard
    if ord('0') <= key <= ord('9'):
        section = key - ord('0')  # press 0, 1, 2, ... 9

    if rval and frame is not None:
        frame = cv2.pyrDown(frame)  # make smaller immediately

        if section == 0:
            pass  # just display the WebCam image

        elif section == 1:
            # get the width and the height and the depth
            h, w, d = frame.shape
            i = int(h / 2)
            j = int(w / 2)

            # slow access
            t = time.time()
            for iterate in range(0, 10000):
                b = frame[i, j, 0]
                g = frame[i, j, 1]
                r = frame[i, j, 2]
            str1 = "B1:%d, G1:%d, R1:%d, time=%.5f" % (b, g, r, time.time() - t)

            # speed up access using NumPy accelerations
            t = time.time()
            for iterate in range(0, 10000):
                b = frame.item(i, j, 0)
                g = frame.item(i, j, 1)
                r = frame.item(i, j, 2)
            str2 = "B2:%d, G2:%d, R2:%d, time=%.5f" % (b, g, r, time.time() - t)

            # set the value
            for x in itertools.product(range(i - 10, i + 10), range(j - 10, j + 10)):
                frame.itemset(x[0], x[1], 0, 255)  # blue value

            cv2.putText(frame, str1, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
            cv2.putText(frame, str2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255])

        elif section == 2:
            # convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("demowin2", gray)

        elif section == 3:
            # convert to HSV and then grab the Hue component
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            singleChannel = cv2.split(hsv)[0]
            cv2.imshow("demowin2", singleChannel)

        elif section == 4:
            # scaling down an image (can speed things up if your frame rate gets too slow)
            small_frame = cv2.pyrDown(frame)
            cv2.imshow("demowin2", small_frame)

        elif section == 5:
            # smoothing with a filter
            kernel = cv2.getGaussianKernel(25, 3)
            smooth_frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow("demowin2", smooth_frame)

        elif section == 6:
            # Requires a single-channel image:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Reduce the image to the edges detected:
            gray = cv2.Canny(gray, 50, 100, 3)  # Arbitrarily chosen parameters. See documentation for the meaning
            cv2.imshow("demowin2", gray)

        elif section == 7:
            # Hough Circles:  http://en.wikipedia.org/wiki/Hough_transform

            # Requires a single-channel image:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Want to smooth it to get less noise
            kernel = cv2.getGaussianKernel(9, 1)
            gray = cv2.filter2D(gray, -1, kernel)

            # Detect the circles in the image
            circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 4, minDist=800,
                                       param1=300, param2=100, minRadius=30, maxRadius=70)
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                # Iterate through the list of circles found by cvHoughCircles()
                for c in circles:
                    # It has the format: [center_x, center_y, radius]
                    # lets draw them
                    center = (c[0], c[1])  # center of the circle
                    rad = c[2]  # radius of the circle
                    cv2.circle(frame, center, rad, (0, 255, 0), 1)

                    # # There's lots of drawing commands you can use!
                    # Here is some c++ code to get you on the right track:
                    # CvFont font;
                    # cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0.0, 1, 8);
                    # cvCircle( frame, cvPoint(cvRound(p[0]),cvRound(p[1])), cvRound(p[2]), CV_RGB(255,0,0), 3, 8, 0 );
                    # cvPutText( frame, "Circle", cvPoint(cvRound(p[0]),cvRound(p[1])), &font, CV_RGB(255,0,0) );

        # we have an image to process
        cv2.putText(frame, "Section = " + str(section), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])
        cv2.imshow("demowin1", frame)

vc.release()
cv2.destroyAllWindows()