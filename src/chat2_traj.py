import cv2
import numpy as np
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
import csv
import os
import tarfile

# Setup xArm
arm = XArmAPI("192.168.1.160")
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
# arm.move_gohome(wait=True)
arm.set_position(x=190, y=-36, z=405, wait=True)

# Constants
CAMERA_WIDTH_PX = 1280
CAMERA_HEIGHT_PX = 720
REAL_WIDTH_MM = 230
REAL_HEIGHT_MM = 145
Z_HEIGHT = 0.15  # meters
ROLL, PITCH, YAW = 180, 0, 180  # degrees

camera_matrix = np.array([[2275.10834345, 0, 1930.73813053],
                          [0, 2275.10834345, 1070.18874506],
                          [0, 0, 1]], dtype="float32")
dist_coeffs = np.array([0.180311899199, -0.419620740236, 0.000593522191604, -0.000263437384568, 0.218466433408], dtype="float32")

def px_to_meters(x, y, warp_shape):
    scale_x = REAL_WIDTH_MM / warp_shape[1] / 1000.0
    scale_y = REAL_HEIGHT_MM / warp_shape[0] / 1000.0
    return x * scale_x, y * scale_y

def generate_trajectory_csv(contours, warped_shape, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["# frequency=250.000000"])
        for contour in contours:
            for point in contour:
                x_px, y_px = point[0]
                x, y = px_to_meters(x_px, y_px, warped_shape)
                writer.writerow([x, y, Z_HEIGHT, np.radians(ROLL), np.radians(PITCH), np.radians(YAW), 0.0])
    base, _ = os.path.splitext(output_path)
    traj_path = base + '.traj'
    # If a .traj already exists, overwrite it
    if os.path.exists(traj_path):
        os.remove(traj_path)
    os.rename(output_path, traj_path)
    
def archive_trajectory(csv_path, tar_path):
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(csv_path, arcname=os.path.basename(csv_path))

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

# Setup camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH_PX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT_PX)

# AprilTag detector
detector = Detector(families='tag16h5')
os.makedirs("trajectories", exist_ok=True)

cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        print("Exiting loop.")
        break

    elif key == 13:  # ENTER key pressed
        print("Enter pressed: scanning and generating trajectory...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undist = cv2.undistort(gray, camera_matrix, dist_coeffs)
        detections = detector.detect(undist)
        id_map = {det.tag_id: det.center for det in detections}

        if not all(k in id_map for k in [1, 2, 3, 4]):
            print("Not all 4 AprilTags detected.")
            continue

        pts = np.array([id_map[i] for i in [1, 2, 3, 4]], dtype="float32")
        ordered = order_points(pts)

        # Warp image
        tl, tr, br, bl = ordered
        width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        TARGET_LEN = 500
        if width >= height:
            dstW = TARGET_LEN
            dstH = int(TARGET_LEN * height / width)
        else:
            dstH = TARGET_LEN
            dstW = int(TARGET_LEN * width / height)

        dst_pts = np.array([[0, 0], [dstW-1, 0], [dstW-1, dstH-1], [0, dstH-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst_pts)
        warped = cv2.warpPerspective(undist, M, (dstW, dstH))

        # Preprocess for contour detection
        # blurred = cv2.GaussianBlur(warped, (5, 5), 0)
        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # kernel = np.ones((3, 3), np.uint8)
        # dilated = cv2.dilate(thresh, kernel, iterations=2)
        # eroded = cv2.erode(dilated, kernel, iterations=1)
        contours, _ = cv2.findContours(warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(warped, contours, -1, (0, 255, 0), 2)
        # cv2.imshow('result', warped)
        # cv2.waitKey(3000)

        # Save trajectory
        csv_path = "trajectories/contour_traj.csv"
        tar_path = "trajectories/contour_traj.tar.gz"
        generate_trajectory_csv(contours, warped.shape, csv_path)
        csv_path = "trajectories/contour_traj.traj"
        archive_trajectory(csv_path, tar_path)

        print("Executing trajectory...")
        load_ret = arm.load_trajectory(tar_path)
        print("Load result:", load_ret)

        # Then play it
        if load_ret == 0:
            ret = arm.playback_trajectory(filename=tar_path, times=1, wait=True, double_speed=False, speed=1.0)
            print("Playback result:", ret)
        else:
            print("Failed to load trajectory")
        # ret = arm.playback_trajectory(tar_path, wait=True, double_speed=False, speed=1.0)
        # ret = arm.playback_trajectory(filename=tar_path, times=1, wait=True, double_speed=False, speed=1.0)
        # print(f"Trajectory execution result: {ret}")

cap.release()
cv2.destroyAllWindows()
