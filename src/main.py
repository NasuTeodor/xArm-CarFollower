import cv2
from pupil_apriltags import Detector
import numpy as np

# def order_points(pts):
#     # Convert to NumPy array of shape (4,2)
#     pts = np.array(pts, dtype="float32")
#     # The top-left point has the smallest sum, bottom-right has the largest sum
#     s = pts.sum(axis=1)
#     top_left = pts[np.argmin(s)]
#     bottom_right = pts[np.argmax(s)]
#     # The top-right has the smallest difference (y - x), bottom-left has the largest difference
#     diff = np.diff(pts, axis=1).flatten()
#     top_right = pts[np.argmin(diff)]
#     bottom_left = pts[np.argmax(diff)]
#     return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = Detector(
    families='tag16h5',
    nthreads=1
)

# Camera intrinsics
fx=2275.10834345
fy=2275.10834345
cx=1930.73813053
cy=1070.18874506
k1=0.180311899199
k2=-0.419620740236
p1=0.000593522191604
p2=-0.000263437384568
k3=0.218466433408

# Your camera matrix and distortion coefficients
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # typical
tag_size = 0.029

tag_frames = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Undistort first (optional if you want raw detection then warp)
    undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs)

    detected_ids = {}
    apriltags = detector.detect(undistorted)
    for detection in apriltags:
        tag_id = detection.tag_id
        position = detection.center
        if detection.decision_margin > 0.2:
            try:
                tag_frames[tag_id] += 1
            except:
                tag_frames[tag_id] = 0
            if tag_frames[tag_id] >= 20:
                detected_ids[tag_id] = position
        else:
            try:
                tag_frames[tag_id] = max(tag_frames[tag_id] - 1, 0)
            except:
                pass

    for tag_id in list(tag_frames.keys()):
        if tag_frames[tag_id] > 0 and not any(detection.tag_id == tag_id for detection in apriltags):
            tag_frames[tag_id] -= 1

    corners = []
    for i in range(1, 5):
        try:
            corners.append(detected_ids[i])
        except:
            pass
            
    if len(corners) != 4:
        continue

    frame_copy = undistorted.copy()
    for tag in corners:
        pose = (int(tag[0]), int(tag[1]))
        cv2.circle(frame_copy, pose, 1, (0,0,0), 8)
        
    corners = np.array(corners)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).flatten()

    ordered_points = np.zeros((4, 2), dtype="float32")
    ordered_points[0] = corners[np.argmin(sums)]  # top-left
    ordered_points[2] = corners[np.argmax(sums)]  # bottom-right
    ordered_points[1] = corners[np.argmin(diffs)] # top-right
    ordered_points[3] = corners[np.argmax(diffs)] # bottom-left

    square_size = 500  # or whatever you want
    dst_pts = np.array([
        [0, 0],
        [square_size-1, 0],
        [square_size-1, square_size-1],
        [0, square_size-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_points, dst_pts)
    # # Warp the undistorted image
    warped = cv2.warpPerspective(undistorted, M, (square_size, square_size))

    cv2.imshow('Warped', warped)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()