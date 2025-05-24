import cv2
import numpy as np
import math
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
import time

# ———— Configurare xArm ————
arm = XArmAPI("192.168.1.160")
arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

Z_MASINA = 1.4

arm.open_lite6_gripper()
arm.set_position(x=267, y=228, z=68, wait=True, roll=180, pitch=0, yaw=0)



arm.set_position(x=267, y=228, z=Z_MASINA, wait=True, roll=180, pitch=0, yaw=0)
time.sleep(1)

arm.close_lite6_gripper()
time.sleep(1)

arm.set_position(x=190, y=-36, z=405, wait=True, roll=180, pitch=0, yaw=0)

# ———— Constante ————
CAM_W, CAM_H = 1280, 720
REAL_W, REAL_H = 230.0, 145.0  # Verifică dimensiunile reale ale foii
ORIGIN = (150.0, 0.0)
X_OFFSET = -35.0
Y_OFFSET = -70.0
Z_SAFE, Z_DRAW = 100, 18
SPEED_T, SPEED_D = 100, 100
MIN_DIST_PX = 15
MARGIN = 20

# ———— Intrinseci cameră ————
camera_matrix = np.array([[2275.10834345, 0, 1930.73813053],
                          [0, 2275.10834345, 1070.18874506],
                          [0, 0, 1]], dtype="float32")
dist_coeffs = np.array([0.180311899199, -0.419620740236,
                        0.000593522191604, -0.000263437384568,
                        0.218466433408], dtype="float32")

# ———— Funcții ajutătoare ————
def px_to_robot(pt, dstW, dstH):
    x_px, y_px = pt
    full_x = x_px + MARGIN
    full_y = y_px + MARGIN
    SCALE_X = REAL_W / (dstW - 2 * MARGIN)
    SCALE_Y = REAL_H / (dstH - 2 * MARGIN)
    x_mm = ORIGIN[0] + full_x * SCALE_X + X_OFFSET
    y_mm = ORIGIN[1] + full_y * SCALE_Y + Y_OFFSET
    print(f"Pixel ({x_px}, {y_px}) -> Robot ({x_mm:.2f}, {y_mm:.2f})")
    return x_mm, y_mm

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

def simplify_contour(cnt):
    out, prev = [], None
    for p in cnt:
        x, y = p[0]
        if prev is None or (x - prev[0])**2 + (y - prev[1])**2 >= MIN_DIST_PX**2:
            out.append((x, y))
            prev = (x, y)
    return out if len(out) > 1 else []

def generate_gcode(simplified_list, dstW, dstH):
    gcode = ["G21", "G90"]
    for simp in simplified_list:
        x0, y0 = px_to_robot(simp[0], dstW, dstH)
        gcode.append(f"G01 X{x0:.2f} Y{y0:.2f} Z{Z_SAFE} F{SPEED_T}")
        gcode.append(f"G01 X{x0:.2f} Y{y0:.2f} Z{Z_DRAW} F{SPEED_D}")
        for i in range(1, len(simp)):
            x1, y1 = px_to_robot(simp[i-1], dstW, dstH)
            x2, y2 = px_to_robot(simp[i], dstW, dstH)
            # Calculăm unghiul tangentei
            yaw = math.degrees(math.atan2(y2 - y1, x2 - x1))
            print(f"Yaw la punctul {i}: {yaw:.2f} grade")
            # Mișcare cu yaw ajustat
            gcode.append(f"G01 X{x2:.2f} Y{y2:.2f} Z{Z_DRAW} F{SPEED_D}")
            # În loc de set_servo_angle, folosim set_position cu yaw
            # arm.set_position(x=x2, y=y2, z=Z_DRAW, yaw=yaw, wait=True, speed=SPEED_D)
        gcode.append(f"G01 Z{Z_SAFE} F{SPEED_T}")
    return gcode

# ———— Configurare video și detector ————
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
detector = Detector(families="tag16h5")

cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Warped ROI", cv2.WINDOW_NORMAL)
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.namedWindow("Skeleton", cv2.WINDOW_NORMAL)  # Fereastră pentru schelet

is_moving = False
simplified_list = []

while True:
    
    
    
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(gray, camera_matrix, dist_coeffs)
    dets = detector.detect(undist)
    id_map = {d.tag_id: d.center for d in dets}

    if all(i in id_map for i in (1, 2, 3, 4)):
        pts = [id_map[i] for i in (1, 2, 3, 4)]
        ordered = order_points(pts)
        tl, tr, br, bl = ordered
        w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        TARGET = 500
        if w >= h:
            dstW, dstH = TARGET, int(TARGET * h / w)
        else:
            dstH, dstW = TARGET, int(TARGET * w / h)
        dst_pts = np.array([[dstW-1, dstH-1], [dstW-1, 0], [0, 0], [0, dstH-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst_pts)
        warped_full = cv2.warpPerspective(undist, M, (dstW, dstH))
        warped = warped_full[MARGIN:dstH-MARGIN, MARGIN:dstW-MARGIN]

        # Procesare traiectorie cu scheletizare îmbunătățită
        blur = cv2.GaussianBlur(warped, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ker = np.ones((5, 5), np.uint8)  # Mărire kernel pentru linii groase
        ero = cv2.erode(cv2.dilate(th, ker, iterations=3), ker, iterations=2)
        skeleton = cv2.ximgproc.thinning(ero)
        cnts, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrăm pentru a păstra doar conturul cel mai lung
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]  # Păstrează doar cel mai mare contur
        simplified_list = []
        overlay = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        for c in cnts:
            simp = simplify_contour(c)
            if simp:
                simplified_list.append(simp)
                for i in range(len(simp)-1):
                    p1 = tuple(map(int, simp[i]))
                    p2 = tuple(map(int, simp[i+1]))
                    cv2.line(overlay, p1, p2, (0, 255, 0), 2)

        # Afișăm scheletul pentru depanare
        skeleton_display = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Skeleton", skeleton_display)

    cv2.imshow("Live Feed", frame)
    if dstW and dstH:
        cv2.imshow("Warped ROI", warped)
        cv2.imshow("Contours", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        break
    if key == 13 and simplified_list and not is_moving:  # Enter
        is_moving = True
        gcode = generate_gcode(simplified_list, dstW, dstH)
        for cmd in gcode:
            arm.send_cmd_sync(cmd)
        is_moving = False
        print("Raliu complet!")

cap.release()
cv2.destroyAllWindows()
arm.set_position(x=267, y=228, z=45, wait=True, roll=180, pitch=0, yaw=0)
arm.set_position(x=267, y=228, z=Z_MASINA, wait=True, roll=180, pitch=0, yaw=0)
time.sleep(1)

arm.open_lite6_gripper()
time.sleep(1)

arm.set_position(x=267, y=228, z=45, wait=True, roll=180, pitch=0, yaw=0)
arm.move_gohome(wait=True)
arm.motion_enable(enable=False)