import cv2
import numpy as np


def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("left click")
        circle.append((x,y))


cap = cv2.VideoCapture(0)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_drawing)
circle= []
while True: 
    _, frame = cap.read()
    for center_position in circle:
       cv2.circle(frame,center_position, 5 , (0,0,255),-1)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
