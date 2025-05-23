import cv2
import numpy as np
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.1.160")
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

arm.move_gohome(wait=True)

# ———————————————
# Helper: order four points as TL, TR, BR, BL
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


# ———————————————
# Camera setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = Detector(
    families='tag16h5',
    nthreads=1
)

# Intrinsics & distortion coeffs
fx, fy = 2275.10834345, 2275.10834345
cx, cy = 1930.73813053, 1070.18874506
k1, k2 = 0.180311899199, -0.419620740236
p1, p2 = 0.000593522191604, -0.000263437384568
k3 = 0.218466433408

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype="float32")
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype="float32")

# For stability: how many consecutive frames a tag must be seen
MIN_FRAMES = 20
tag_frames = {}

# Target maximum side length of extracted ROI
TARGET_LEN = 500

# ———————————————
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(gray, camera_matrix, dist_coeffs)

    # Detect AprilTags
    detections = detector.detect(undist)
    detected_ids = {}

    # Update frame counters per tag for stability
    for det in detections:
        tid = det.tag_id
        if det.decision_margin < 0.2:
            tag_frames[tid] = max(tag_frames.get(tid, 0) - 1, 0)
        else:
            tag_frames[tid] = tag_frames.get(tid, 0) + 1

        if tag_frames[tid] >= MIN_FRAMES:
            detected_ids[tid] = det.center

    # Decay tags not seen this frame
    for tid in list(tag_frames.keys()):
        if tag_frames[tid] > 0 and not any(d.tag_id == tid for d in detections):
            tag_frames[tid] -= 1

    # Collect the four required corners (IDs 1–4)
    corners = []
    for i in range(1, 5):
        if i in detected_ids:
            corners.append(detected_ids[i])

    if len(corners) != 4:
        # Not enough stable tags yet
        cv2.imshow('frame', undist)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Order them TL, TR, BR, BL
    ordered = order_points(corners)

    # Compute widths and heights of the source quadrilateral
    tl, tr, br, bl = ordered
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxW = max(int(widthA), int(widthB))
    maxH = max(int(heightA), int(heightB))

    # Scale so the longer side == TARGET_LEN
    if maxW >= maxH:
        dstW = TARGET_LEN
        dstH = int(TARGET_LEN * maxH / maxW)
    else:
        dstH = TARGET_LEN
        dstW = int(TARGET_LEN * maxW / maxH)

    # Destination rectangle
    dst_pts = np.array([
        [0,      0],
        [dstW-1, 0],
        [dstW-1, dstH-1],
        [0,      dstH-1]
    ], dtype="float32")

    # Perspective warp
    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(undist, M, (dstW, dstH))

    # Visualization: draw the detected quadrilateral
    vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)
    pts = ordered.astype(int)
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,0), 2)
        cv2.circle(vis, tuple(pts[i]), 5, (0,0,255), -1)

    # Show outputs
    cv2.imshow('Detected Rectangle', vis)
    cv2.imshow('Warped ROI', warped)

    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(warped, (5, 5), 0)

    # Use a threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Increase the size of the kernel
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Detect contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #continue with following each contour

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
