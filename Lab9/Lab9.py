import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from Tello_Video.keyboard_djitellopy import keyboard

fs = cv2.FileStorage('Camera.xml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeff').mat()
fs.release()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.4, kI=0.0, kD=0.2)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

global dx, dy, dz, d, dh, dw
dx = dy = dz = 0.1
d = 0.5
dh = 0
dw = 0

max_speed_threshold = 20
marker_length = 0.15
drone.send_rc_control(0, 0, 0, 0)


def process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs, yraw_d = 5):
    detect = False
    
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # 畫出坐標軸
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        
        # 計算旋轉矩陣並將Z軸向量投影到XZ平面
        R, _ = cv2.Rodrigues(rvec)
        Z_axis = np.dot(R, np.array([0, 0, 1]))
        angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])  # Z 與 XZ 平面夾角

        # 用 math.degrees 轉換成度數
        yaw_degrees = math.degrees(angle_to_marker)
        yaw_degrees += 90

        # 獲取 tvec 的 x, y, z 座標
        x, y, z = tvec[0]
        if (x - dw < dx and x - dw > -1 * dx and y - dh < dy and y - dh > -1 * dy and z - d < dz and z - d > -1 * dz and yaw_degrees < yraw_d and yaw_degrees > (-1 * yraw_d)):
            detect = True
            print(f"{x=}, {y=}, {z=}, {yaw_degrees=}")
            break


        # 構建顯示字串
        text = f"ID: {markerIds[0]}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m  yaw: {yaw_degrees:.2f} degrees"
        corner = markerCorners[i][0][0]
        cv2.putText(frame, text, (int(corner[0]), int(corner[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # 控制無人機移動
        z_update = z_pid.update(50 * (z - d), 0)  # 前進後退
        x_update = x_pid.update(50 * x, 0)  # 左右移動
        y_update = y_pid.update(50 * (y - dh), 0)  # 上下移動
        yaw_update = yaw_pid.update(yaw_degrees, 0)  # 旋轉

        # 限制速度範圍
        z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
        x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
        y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)

        # 使用無人機的控制命令
        if (y_update > 0):
            y_update = y_update * 1.5
        
        drone.send_rc_control(int(1.5 * int(x_update)), int(1.5 * z_update),
                              int(-2.5 * int(y_update)), -1 * int(yaw_update))

    return detect

def follow_line(frame, drone, direction, prev_angle):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        is_corner, current_angle = detect_corner(frame, contours, prev_angle)

        if is_corner:
            print("轉角偵測到！")
            return True, True, current_angle
        
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])  # 計算重心 X 座標
            cy = int(M["m01"] / M["m00"])  # 計算重心 Y 座標

            # 在影像上標記重心
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            dthreshold = frame.shape[1] // 4

            # 設定影像中線條的中心作為目標
            frame_center = frame.shape[1] // 2

            # 根據重心位置控制無人機左右移動
            print(f"{cx=}, {cy=}")
            print(f"{frame_center=}")
            drone.send_rc_control(direction[0], direction[1], direction[2], direction[3])
            print(f"Follow line: ({direction[0]}, {direction[1]}, {direction[2]}, {direction[3]})")
            time.sleep(0.5)
            
            return True, False, current_angle

    return False, False, prev_angle

def detect_corner(frame, contours, prev_angle=None):
    if not contours:
        return False, prev_angle  # 無輪廓，無法判斷
    
    largest_contour = max(contours, key=cv2.contourArea)

    [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    direction_angle = math.degrees(math.atan2(vy.item(), vx.item()))

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) > 4:
        print("偵測到轉角(輪廓複雜)!")
        return True, direction_angle

    return False, direction_angle


while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)

    frame = frame_read.frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)
    if markerIds is not None and 0 in markerIds:
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detected = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
        if (detected): 
            print("偵測到 Marker")
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(20, 0, 0, 0)
            time.sleep(1.5)
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(3)
            break
        
    cv2.imshow("Frame", frame)

direction = [10, 0, 0, 0]
finish = 0
prev_angle = None

while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)

    frame = frame_read.frame
    
    scale_percent = 50  # 縮放比例，50 表示縮小為原來的 50%
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h, w = gray_frame.shape
    start_row = h // 5  # 上方裁剪 1/4
    end_row = start_row * 4  # 中間的 1/2
    start_col = w // 5  # 左右裁剪 1/4
    end_col = start_col * 4  # 中間的 1/2
    
    cropped_frame = gray_frame[start_row:end_row, start_col:end_col]
    # cropped_frame = gray_frame[:, :]
    
    # kernel_size = 5  # 可以根據需求調整
    # cropped_frame = cv2.GaussianBlur(cropped_frame, (kernel_size, kernel_size), 0)

    threshold_value = 127  # 閾值
    max_value = 255  # 超過閾值的像素設為此值
    _, threshold_frame = cv2.threshold(cropped_frame, threshold_value, max_value, cv2.THRESH_BINARY)
    
    dilation_kernel = np.ones((3, 3), np.uint8)  # 定義一個 3x3 的矩形內核
    dilation = cv2.dilate(threshold_frame, dilation_kernel, iterations=1)
    
    line_frame = cv2.erode(dilation, dilation_kernel, iterations=1)

    cv2.imshow("Line Frame", line_frame)
    cv2.imshow("Frame", frame)
    
    if finish == 5:
        print("結束")
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if markerIds is not None and 1 in markerIds:
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

            detected = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
            if detected:
                drone.send_rc_control(0, 0, 0, 0)
                time.sleep(1)
                drone.land()
                break
            continue    # 偵測到 marker 後就不管線了

    line_detected, corner, prev_angle = follow_line(line_frame, drone, direction, prev_angle)

    if corner:
        if finish == 0:
            print("detect corner 1")
            direction = [0, 0, 10, 0]
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(0, 0, 30, 0)
            time.sleep(2)
            drone.send_rc_control(0, 0, 0, 0)
            finish = 1
        elif finish == 1:
            print("detect corner 2")
            direction = [10, 0, 0, 0]
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(15, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(0, 0, 0, 0)
            finish = 2
        elif finish == 2:
            print("detect corner 3")
            direction = [0, 0, 10, 0]
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(0, 0, 30, 0)
            time.sleep(2)
            drone.send_rc_control(0, 0, 0, 0)
            finish = 3
        elif finish == 3:
            print("detect corner 4")
            direction = [-10, 0, 0, 0]
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(-20, 0, 0, 0)
            time.sleep(2)
            drone.send_rc_control(0, 0, 0, 0)
            finish = 4
        elif finish == 4:
            print("detect corner 5")
            direction = [0, 0, -10, 0]
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(0, 0, -20, 0)
            time.sleep(2)
            drone.send_rc_control(0, 0, 0, 0)
            finish = 5
        continue
    
    if line_detected:
        continue
