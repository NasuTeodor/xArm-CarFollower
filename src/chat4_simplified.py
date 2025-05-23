import cv2
import numpy as np
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
import math

# ———— xArm Setup ————
arm = XArmAPI("192.168.1.160")
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
# arm.move_gohome(wait=True)
arm.set_position(x=190, y=-36, z=405, wait=True, roll=180, pitch=0, yaw=0)


# ———— Constants ————
CAM_W_PX, CAM_H_PX = 1280, 720
REAL_W_MM, REAL_H_MM = 230.0, 145.0
# convert pixels → mm
SCALE_X = REAL_W_MM / CAM_W_PX
SCALE_Y = REAL_H_MM / CAM_H_PX
ORIGIN_XY = (200.0, 0.0)    # robot-frame offset, adjust as needed

Z_SAFE = 100   # mm above surface for transit
Z_DRAW = 50    # mm above surface while drawing
SPEED_TRANSIT = 100   # mm/s
SPEED_DRAW    = 100   # mm/s

ROLL_CONST  = 180.0  # degrees
PITCH_CONST =   0.0  # degrees

# ———— Camera Intrinsics ————
camera_matrix = np.array([
    [2275.10834345, 0,             1930.73813053],
    [0,             2275.10834345, 1070.18874506],
    [0,             0,             1           ]
], dtype="float32")
dist_coeffs = np.array([
    0.180311899199, -0.419620740236,
    0.000593522191604, -0.000263437384568,
    0.218466433408
], dtype="float32")

# ———— Helpers ————
def px_to_robot(pt):
    """Convert pixel (x,y) to robot-frame (x_mm, y_mm)."""
    x_px, y_px = pt
    return ORIGIN_XY[0] + x_px * SCALE_X, ORIGIN_XY[1] + y_px * SCALE_Y

def order_points(pts):
    """Order 4 points TL → TR → BR → BL."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")

# ———— Video + Detector Setup ————
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W_PX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H_PX)

detector = Detector(families="tag16h5")

cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

# ———— Main Loop ————
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Esc
        print("Exiting.")
        break

    if key != 13:  # wait for Enter
        continue

    # — Capture on Enter —
    print("Enter pressed: capturing and tracing contours.")
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(gray, camera_matrix, dist_coeffs)

    # --- detect AprilTags 1–4
    dets = detector.detect(undist)
    id_map = {d.tag_id: d.center for d in dets}
    if not all(i in id_map for i in (1,2,3,4)):
        print("AprilTags 1–4 not all found; retry.")
        continue

    pts = [ id_map[i] for i in (1,2,3,4) ]
    ordered = order_points(pts)

    # --- warp to front-view rectangle
    tl, tr, br, bl = ordered
    w = max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))
    h = max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))
    TARGET = 500
    if w >= h:
        dstW, dstH = TARGET, int(TARGET * h / w)
    else:
        dstH, dstW = TARGET, int(TARGET * w / h)

    dst_pts = np.array([[0,0],[dstW-1,0],[dstW-1,dstH-1],[0,dstH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(undist, M, (dstW, dstH))

    # --- preprocess & find contours
    blur = cv2.GaussianBlur(warped, (5,5), 0)
    _,th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ker = np.ones((3,3), np.uint8)
    dil = cv2.dilate(th, ker, iterations=2)
    ero = cv2.erode(dil, ker, iterations=1)
    contours, _ = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- trace each contour
    # … up near the top, define:
    MIN_DIST_PX = 15    # minimum distance (in pixels) between successive points

    # … later, where you trace each contour:
    for cnt in contours:
        if len(cnt) < 3:
            continue

        # Prune too-close points
        simplified = []
        prev = None
        for p in cnt:
            x_px, y_px = p[0]
            if prev is None:
                simplified.append((x_px, y_px))
                prev = (x_px, y_px)
            else:
                dx = x_px - prev[0]
                dy = y_px - prev[1]
                if dx*dx + dy*dy >= MIN_DIST_PX**2:
                    simplified.append((x_px, y_px))
                    prev = (x_px, y_px)
        if len(simplified) < 2:
            continue

        # Move above start
        x0_mm, y0_mm = px_to_robot(simplified[0])
        arm.set_position(x=x0_mm, y=y0_mm, z=Z_SAFE,
                         wait=True, speed=SPEED_TRANSIT)

        # Lower to draw height
        arm.set_position(x=x0_mm, y=y0_mm, z=Z_DRAW,
                         wait=True, speed=SPEED_DRAW)

        # Trace with translation + wrist-only rotation
        for i in range(len(simplified)-1):
            x_px, y_px   = simplified[i]
            x2_px, y2_px = simplified[i+1]
            x_mm, y_mm   = px_to_robot((x_px, y_px))
            x2_mm, y2_mm = px_to_robot((x2_px, y2_px))

            # Translate
            arm.set_position(x=x_mm, y=y_mm, z=Z_DRAW,
                             wait=True, speed=SPEED_DRAW)

            # Compute yaw tangent
            angle_rad = math.atan2(y2_mm - y_mm, x2_mm - x_mm)
            yaw = math.degrees(angle_rad)
            # leave yaw in –180…+180

            # Only rotate wrist joint (joint 6)
            resp = arm.get_servo_angle()
            if resp[0] == 0:
                angles = resp[1]
                angles[5] = yaw  # Update joint 6 angle
                arm.set_servo_angle(angle=angles, is_radian=False, wait=True, speed=500, mvaccel=500)
            else:
                print("Warning: failed to read joint angles")

        # Draw last point
        x_last_mm, y_last_mm = px_to_robot(simplified[-1])
        arm.set_position(x=x_last_mm, y=y_last_mm, z=Z_DRAW,
                         wait=True, speed=SPEED_DRAW)

        # Lift after contour
        arm.set_position(z=Z_SAFE, relative=True,
                         wait=True, speed=SPEED_TRANSIT)

    print("Trace complete; waiting for next Enter.")

cap.release()
cv2.destroyAllWindows()
# arm.move_gohome(wait=True)