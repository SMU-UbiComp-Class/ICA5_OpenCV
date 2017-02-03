import cv2

vc = cv2.VideoCapture(0)

if vc.isOpened():
    print "vc is open!!"
    return_value, frame = vc.read()

h_range = 30
key = -1
while key != 27 and vc.isOpened():
    return_value, frame = vc.read()

    if frame is not None:

        height, width, depth = frame.shape

        frame = cv2.pyrDown(frame)

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)

        h = cv2.inRange(h, h_range, h_range+10)
        h_binary = h.copy()
        _, contours, hierarchy = cv2.findContours(h_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours is not None and len(contours) > 0:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # bb   = cv2.boundingRect(cnt)

                pt = tuple(cnt[0][0])
                if area > 100:
                    cv2.putText(h, str(area), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 255)

        cv2.putText(h, str(h_range), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 128)
        cv2.imshow("WebCam", frame)
        cv2.imshow("Hue", h)

    key = cv2.waitKey(10)
    if key == ord('u'):
        h_range += 1
    elif key == ord('d'):
        h_range -= 1
