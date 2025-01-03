import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from Tello_Video.keyboard_djitellopy import keyboard

# 從 calibration.xml 讀取相機標定參數
fs = cv2.FileStorage('Camera.xml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeff').mat()
fs.release()

# 設定 ArUco 標記字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# 初始化偵測器參數
parameters = cv2.aruco.DetectorParameters()

drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

# PID 控制器設置 (用於控制無人機的旋轉和前後移動)
x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.4, kI=0.0, kD=0.2)  # 控制無人機的旋轉

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()
global finish
finish = 0
global p
p = 0
global dx, dy, dz
dx = dy = dz = 0.15

# 設定目標距離 (無人機到標記的理想距離)
global d, dh, dw
d = 0.5
dh = 0
dw = 0

global cyaw
cyaw = 1
max_speed_threshold = 20  # 最大速度限制
drone.send_rc_control(0, 0, 0, 0)
zero = 0


def process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs, yraw_d = 10):
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
            print(f"{finish=}----------------------------------------")
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
                              int(-2.5 * int(y_update)), -1 * int(cyaw * yaw_update))

    return detect

drone.send_rc_control(0, 0, 0, 0)

while True:
    key = cv2.waitKey(1)
    if key == ord('f'):
        drone.send_rc_control(0, 80, 0, 0)
        time.sleep(0.5)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
    
        break
    elif key != -1:
        keyboard(drone, key)
    # else:
    #     drone.send_rc_control(0,0,0,0)
    
    frame = frame_read.frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Marker Detection and Pose Estimation', frame)


while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)

    frame = frame_read.frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 偵測 ArUco 標記
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)
    # print(f"{markerIds=},\n {markerCorners=},\n {rejectedCandidates=}")
    if markerIds is not None and 1 in markerIds and finish == 0:
        zero = 1
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 1)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        print(markerIds)
        markerCorners = [markerCorners[i] for i in valid_indices]
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
        if (detect1): 
            drone.send_rc_control(80, 0, 0, 0)
            time.sleep(0.6)
            finish = 1


    elif markerIds is not None and 2 in markerIds and finish == 1:
        zero = 1
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 2)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        markerCorners = [markerCorners[i] for i in valid_indices]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
        if (detect1): 
            drone.send_rc_control(-80, 0, 0, 0)
            time.sleep(0.6)
            finish = 2

    elif markerIds is not None and 3 in markerIds and finish == 2:
        zero = 1
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 3)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        markerCorners = [markerCorners[i] for i in valid_indices]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
        if detect1:
            drone.send_rc_control(0, 0, -60, 0)
            time.sleep(1.2)
            drone.send_rc_control(0, 100, 0, 0)
            time.sleep(2)
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.send_rc_control(0, 0, 60, 0)
            time.sleep(1.5)
            finish = 3

    elif markerIds is not None and 0 in markerIds and finish == 3:
        zero = 1
        cyaw = 1.5
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 0)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        markerCorners = [markerCorners[i] for i in valid_indices]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)

    elif markerIds is not None and 4 in markerIds and finish == 3:
        d = 0.6
        zero = 1
        detect1 = 0
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 4)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        markerCorners = [markerCorners[i] for i in valid_indices]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
        if detect1:
            drone.send_rc_control(0, 0, 0, 90)
            time.sleep(1)
            finish = 4

    elif markerIds is not None and 5 in markerIds and finish == 4:
        d = 0.6
        zero = 1
        detect1 = 0
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 5)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        markerCorners = [markerCorners[i] for i in valid_indices]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs)
        if detect1:
            drone.send_rc_control(-100, 0, 0, 0)
            time.sleep(1.5)
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1.5)
            drone.send_rc_control(0, -20, 0, 0)
            time.sleep(2)
            # drone.send_rc_control(-100, 0, 0, 0)
            # time.sleep(1)
            finish = 5

    elif markerIds is not None and 6 in markerIds and finish == 5:
        zero = 1
        detect1 = 0
        detect2 = 0
        print(f'{markerIds=}')
        # 過濾出 id == 1 的標記
        ids_array = np.array(markerIds).flatten()
        valid_indices = np.where(ids_array == 6)[0]  # 取得符合條件的索引
        markerIds = ids_array[valid_indices]
        markerCorners = [markerCorners[i] for i in valid_indices]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # 計算每個標記的姿態
        marker_length = 0.15
        d = 1.5
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        detect1 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs, 5)
        if detect1:
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            dx = 0.1
            dh = -0.35
            dw = -0.1
            detect2 = process_aruco_markers(markerIds, markerCorners, rvecs, tvecs, d, z_pid, x_pid, y_pid, yaw_pid, max_speed_threshold, drone, camera_matrix, dist_coeffs, 5)
            if detect2:
                drone.send_rc_control(0, 0, 0, 0)
                time.sleep(1)
                drone.land()
                finish = 6

    else:
        if zero != 0:
            zero = 0
            drone.send_rc_control(0, 0, 0, 0)

    cv2.imshow('Marker Detection and Pose Estimation', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cv2.destroyAllWindows()
drone.streamoff()