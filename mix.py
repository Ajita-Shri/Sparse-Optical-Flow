import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0 )
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = (0,0,255)

#Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)
cv.namedWindow("Frame")
cv.setMouseCallback("Frame", select_point)
point_selected = False
point = ()
old_points = np.array([[]])

mask = np.zeros_like(old_frame)

while(True):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# calculate optical flow
    if point_selected is True:

        cv.circle(frame, point, 5, (0, 0, 255), 2)

        p1, st, error = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)
# Select good points
        good_new = p1[st==1]
        good_old = old_points[st==1]
# draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()

            c,d = old.ravel()

            mask = cv.line(mask, (a,b),(c,d), color, 2)

            frame = cv.circle(frame,(a,b),5,color,-1)

        img = cv.add(frame,mask)
        old_gray = frame_gray.copy()

        old_points = good_new.reshape(-1, 1, 2)

        cv.imshow('image',img)

        k = cv.waitKey(30) & 0xff

        if k == 27:

            break



    # Now update the previous frame and previous points


cv.destroyAllWindows()
cap.release()

