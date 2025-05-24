import cv2
import numpy as np
import math
import time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

# ———— Configurare xArm ————
try:
    arm = XArmAPI("192.168.1.160")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
except Exception as e:
    print(f"Eroare la inițializarea xArm: {e}")
    exit(1)

Z_MASINA = 1.3

# Poziție inițială
try:
    arm.set_position(x=190, y=-36, z=405, wait=True, roll=180, pitch=0, yaw=0)
except Exception as e:
    print(f"Eroare la setarea poziției inițiale: {e}")

# ———— Constante ————
CAM_W, CAM_H = 1280, 720
REAL_W, REAL_H = 230.0, 145.0
ORIGIN = (150.0, 0.0)
X_OFFSET = 20
Y_OFFSET = -70.0
Z_SAFE, Z_DRAW = 100, 18
SPEED_T, SPEED_D = 100, 100
MIN_DIST_PX = 15
MARGIN = 20
MIN_CONTOUR_LENGTH = 30  # Redus pentru a permite contururi mai scurte

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
        arm.set_position(x=x0, y=y0, z=Z_SAFE, roll=180, pitch=0, yaw=0, wait=True, speed=SPEED_T)
        gcode.append(f"G01 X{x0:.2f} Y{y0:.2f} Z{Z_DRAW} F{SPEED_D}")
        arm.set_position(x=x0, y=y0, z=Z_DRAW, roll=180, pitch=0, yaw=0, wait=True, speed=SPEED_D)
        for i in range(1, len(simp)):
            x1, y1 = px_to_robot(simp[i-1], dstW, dstH)
            x2, y2 = px_to_robot(simp[i], dstW, dstH)
            # Calculăm unghiul tangentei
            yaw = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Normalizăm yaw între -180 și 180
            yaw = ((yaw + 180) % 360) - 180
            print(f"Punct {i}: X={x2:.2f}, Y={y2:.2f}, Yaw={yaw:.2f} grade")
            # Mișcare cu yaw ajustat
            gcode.append(f"G01 X{x2:.2f} Y{y2:.2f} Z{Z_DRAW} F{SPEED_D}")
            # Aplicăm yaw-ul folosind set_position
            try:
                resp = arm.set_position(x=x2, y=y2, z=Z_DRAW, roll=180, pitch=0, yaw=yaw, wait=True, speed=SPEED_D)
                print(f"Set position response: {resp}")
            except Exception as e:
                print(f"Eroare la setarea poziției cu yaw: {e}")
            # Alternativă: set_servo_angle pentru joint 6 (comentează dacă funcționează set_position)
            # try:
            #     resp = arm.set_servo_angle(servo_id=6, angle=yaw, is_radian=False, wait=True)
            #     print(f"Set servo angle response: {resp}")
            # except Exception as e:
            #     print(f"Eroare la setarea servo 6: {e}")
        gcode.append(f"G01 Z{Z_SAFE} F{SPEED_T}")
        arm.set_position(x=x2, y=y2, z=Z_SAFE, roll=180, pitch=0, yaw=0, wait=True, speed=SPEED_T)
    return gcode

# ———— Configurare video și detector ————
try:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
except Exception as e:
    print(f"Eroare la inițializarea camerei: {e}")
    exit(1)

detector = Detector(families="tag16h5")

cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Warped ROI", cv2.WINDOW_NORMAL)
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.namedWindow("Skeleton", cv2.WINDOW_NORMAL)

is_moving = False
simplified_list = []

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Eroare: Nu se poate citi cadrul de la cameră")
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

            # Procesare traiectorie
            blur = cv2.GaussianBlur(warped, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ker = np.ones((7, 7), np.uint8)
            ero = cv2.erode(cv2.dilate(th, ker, iterations=4), ker, iterations=3)
            skeleton = cv2.ximgproc.thinning(ero)
            cnts, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            simplified_list = []
            overlay = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            if cnts:
                cnts = sorted(cnts, key=lambda c: len(simplify_contour(c)), reverse=True)[:1]
                print(f"Număr contururi detectate: {len(cnts)}")
                for i, c in enumerate(cnts):
                    simp = simplify_contour(c)
                    if len(simp) >= MIN_CONTOUR_LENGTH:
                        simplified_list.append(simp)
                        print(f"Contur {i}: {len(simp)} puncte")
                        for j in range(len(simp)-1):
                            p1 = tuple(map(int, simp[j]))
                            p2 = tuple(map(int, simp[j+1]))
                            cv2.line(overlay, p1, p2, (0, 255, 0), 2)
            print(f"Număr contururi simplificate: {len(simplified_list)}")

            skeleton_display = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Skeleton", skeleton_display)

        cv2.imshow("Live Feed", frame)
        if 'dstW' in locals() and 'dstH' in locals():
            cv2.imshow("Warped ROI", warped)
            cv2.imshow("Contours", overlay)

        key = cv2.waitKey(1) & 0xFF
        # print(f"Tastă apăsată: {key}")  # Depanare input tastatură
        if key == 27:  # Esc
            break
        if key == 13 and simplified_list and not is_moving:  # Enter
            print("Începe execuția traiectoriei!")
            # Preluare mașinuță
            try:
                arm.open_lite6_gripper()
                arm.set_position(x=267, y=228, z=68, wait=True, roll=180, pitch=0, yaw=0)
                arm.set_position(x=267, y=228, z=Z_MASINA, wait=True, roll=180, pitch=0, yaw=0)
                time.sleep(1)
                arm.close_lite6_gripper()
                time.sleep(1)
                arm.set_position(x=267, y=228, z=68, wait=True, roll=180, pitch=0, yaw=0)
            except Exception as e:
                print(f"Eroare la preluarea mașinuței: {e}")

            is_moving = True
            try:
                gcode = generate_gcode(simplified_list, dstW, dstH)
                for cmd in gcode:
                    resp = arm.send_cmd_sync(cmd)
                    print(f"G-code command response: {resp}")
            except Exception as e:
                print(f"Eroare la execuția G-code: {e}")
            is_moving = False
            print("Raliu complet!")

    except Exception as e:
        print(f"Eroare în bucla principală: {e}")
        break

# Eliberare mașinuță
try:
    arm.set_position(x=267, y=228, z=68, wait=True, roll=180, pitch=0, yaw=0)
    arm.set_position(x=267, y=228, z=Z_MASINA, wait=True, roll=180, pitch=0, yaw=0)
    time.sleep(1)
    arm.open_lite6_gripper()
    time.sleep(1)
    arm.set_position(x=267, y=228, z=68, wait=True, roll=180, pitch=0, yaw=0)
    arm.move_gohome(wait=True)
    arm.motion_enable(enable=False)
except Exception as e:
    print(f"Eroare la eliberarea mașinuței: {e}")

cap.release()
cv2.destroyAllWindows()