import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Take first frame
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
color = (0, 255, 0)


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
# create a mask image
mask = np.zeros_like(old_gray)

while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    if point_selected is True:
        cv.circle(frame, point, 5, (0, 0, 255), 2)
        new_points, st, error = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

        # to draw tracks
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(frame, (a, b), (c, d), color, 2)
            frame = cv.circle(frame, (a, b), 5, color, -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        old_gray = frame_gray.copy()
        old_points = new_points.reshape(-1, 1, 2)

    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
cv.destroyAllWindows()
cap.release()
