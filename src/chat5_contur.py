import cv2
import numpy as np
import math
import threading
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

# ———— xArm Setup ————
arm = XArmAPI("192.168.1.160")
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
arm.set_position(x=190, y=-36, z=405, wait=True, roll=180, pitch=0, yaw=0)

# ———— Constants ————
CAM_W, CAM_H = 1280, 720
REAL_W, REAL_H = 230.0, 145.0
SCALE_X = REAL_W / CAM_W
SCALE_Y = REAL_H / CAM_H
ORIGIN = (200.0, 0.0)

Z_SAFE, Z_DRAW = 100, 50
SPEED_T, SPEED_D = 100, 100
MIN_DIST_PX = 15
MARGIN = 20  # pixels to inset the warped ROI

# ———— Camera Intrinsics ————
camera_matrix = np.array([[2275.10834345, 0, 1930.73813053],
                          [0, 2275.10834345, 1070.18874506],
                          [0, 0, 1]], dtype="float32")
dist_coeffs = np.array([0.180311899199, -0.419620740236,
                        0.000593522191604, -0.000263437384568,
                        0.218466433408], dtype="float32")

# ———— Helpers ————
def px_to_robot(pt, dstW, dstH):
    # pt is in cropped ROI coords
    x_px, y_px = pt
    # shift into full warped coords
    full_x = x_px + MARGIN
    full_y = y_px + MARGIN
    # pixel→mm
    x_mm = ORIGIN[0] + full_x * SCALE_X
    y_mm = ORIGIN[1] + full_y * SCALE_Y
    return x_mm, y_mm

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def simplify_contour(cnt):
    out, prev = [], None
    for p in cnt:
        x, y = p[0]
        if prev is None or (x - prev[0])**2 + (y - prev[1])**2 >= MIN_DIST_PX**2:
            out.append((x, y))
            prev = (x, y)
    return out if len(out) > 1 else []

# ———— Robot tracing thread ————
is_moving = False

def trace_contours(simplified_list, dstW, dstH):
    global is_moving
    is_moving = True
    for simp in simplified_list:
        # move above start
        x0, y0 = px_to_robot(simp[0], dstW, dstH)
        arm.set_position(x=x0, y=y0, z=Z_SAFE, wait=True, speed=SPEED_T)
        arm.set_position(x=x0, y=y0, z=Z_DRAW, wait=True, speed=SPEED_D)
        # trace
        for i in range(len(simp) - 1):
            x_px, y_px   = simp[i]
            x2_px, y2_px = simp[i+1]
            x, y     = px_to_robot((x_px, y_px), dstW, dstH)
            x2, y2   = px_to_robot((x2_px, y2_px), dstW, dstH)
            arm.set_position(x=x, y=y, z=Z_DRAW, wait=True, speed=SPEED_D)
            yaw = math.degrees(math.atan2(y2 - y, x2 - x))
            resp = arm.get_servo_angle()
            if resp[0] == 0:
                arm.set_servo_angle(servo_id=6, angle=yaw, is_radian=False, wait=True)
        # final point & lift
        xl, yl = px_to_robot(simp[-1], dstW, dstH)
        arm.set_position(x=xl, y=yl, z=Z_DRAW, wait=True, speed=SPEED_D)
        arm.set_position(z=Z_SAFE, relative=True, wait=True, speed=SPEED_T)
    is_moving = False
    print("Trace complete.")

# ———— Video + Detector Setup ————
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
detector = Detector(families="tag16h5")

cv2.namedWindow("Live Feed",  cv2.WINDOW_NORMAL)
cv2.namedWindow("Warped ROI", cv2.WINDOW_NORMAL)
cv2.namedWindow("Contours",   cv2.WINDOW_NORMAL)

warped = np.zeros((CAM_H, CAM_W), dtype=np.uint8)
overlay = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
simplified_list = []
dstW = dstH = 0  # will be set each frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort and detect tags
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(gray, camera_matrix, dist_coeffs)
    dets   = detector.detect(undist)
    id_map = {d.tag_id: d.center for d in dets}

    if all(i in id_map for i in (1,2,3,4)):
        pts = [id_map[i] for i in (1,2,3,4)]
        ordered = order_points(pts)
        tl, tr, br, bl = ordered
        w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        TARGET = 500
        if w >= h:
            dstW, dstH = TARGET, int(TARGET * h / w)
        else:
            dstH, dstW = TARGET, int(TARGET * w / h)
        dst_pts = np.array([[0,0],[dstW-1,0],[dstW-1,dstH-1],[0,dstH-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst_pts)
        warped_full = cv2.warpPerspective(undist, M, (dstW, dstH))

        # Crop margin
        warped = warped_full[MARGIN:dstH-MARGIN, MARGIN:dstW-MARGIN]

        # Contour processing on cropped ROI
        blur = cv2.GaussianBlur(warped, (5,5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ker = np.ones((3,3), np.uint8)
        ero = cv2.erode(cv2.dilate(th, ker, 2), ker, 1)
        cnts, _ = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Simplify and overlay
        simplified_list = []
        overlay = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        for c in cnts:
            simp = simplify_contour(c)
            if not simp:
                continue
            simplified_list.append(simp)
            for i in range(len(simp)-1):
                p1 = tuple(map(int, simp[i]))
                p2 = tuple(map(int, simp[i+1]))
                cv2.line(overlay, p1, p2, (0,255,0), 2)

    # Display all windows
    cv2.imshow("Live Feed", frame)
    if dstW and dstH:
        cv2.imshow("Warped ROI", warped)
        cv2.imshow("Contours",   overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        break
    if key == 13 and simplified_list and not is_moving:
        threading.Thread(target=trace_contours,
                         args=(simplified_list, dstW, dstH),
                         daemon=True).start()

cap.release()
cv2.destroyAllWindows()
arm.move_gohome(wait=True)
