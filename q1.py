import numpy as np
import cv2

vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame

    ret, frame = vid.read()
    # cv2.imshow('frame', frame)
    cv2.goodFeaturesToTrack()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 0, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),3,255,-1)
    cv2.imshow("frame",frame)

    if ret == False:
        break
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
