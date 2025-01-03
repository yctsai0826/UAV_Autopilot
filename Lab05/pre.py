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
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

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

# 設定目標距離 (無人機到標記的理想距離)
target_distance = 150  # 設定為毫米
max_speed_threshold = 50  # 最大速度限制

def clamp(value, min_value=-max_speed_threshold, max_value=max_speed_threshold):
    return np.clip(value, min_value, max_value)

while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)
        
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 偵測 ArUco 標記
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

    # 畫出偵測到的標記
    if markerIds is not None:
        print(f'{markerIds=}')
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # 計算每個標記的姿態 (假設標記的邊長為 0.15 米)
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # 計算旋轉矩陣並提取 Yaw
            rotM = np.zeros(shape=(3, 3))
            cv2.Rodrigues(rvec[i], rotM)
            ypr = cv2.RQDecomp3x3(rotM)[0]
            yaw_update = ypr[1] * 1.2  # 調整 yaw

            # 獲取 tvec 的 x, y, z 座標，並計算相對距離
            x, y, z = tvec[0]
            x_update = x
            y_update = y
            z_update = z - target_distance

            # PID 更新並限制速度
            x_update = clamp(x_pid.update(x_update, sleep=0))
            y_update = clamp(y_pid.update(y_update, sleep=0))
            z_update = clamp(z_pid.update(z_update, sleep=0))
            yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))

            print(x_update, y_update, z_update, yaw_update)
            
            # 使用 send_rc_control 控制無人機的移動與旋轉
            drone.send_rc_control(int(x_update), int(z_update // 2), int(y_update), int(yaw_update))

            # 繪製標記及其坐標軸
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            frame = cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 10)

            # 顯示無人機的相對 x, y, z 位置
            cv2.putText(frame, "x = "+str(round(tvec[0, 0, 0], 2)) + ", y = " + str(round(tvec[0, 0, 1], 2)) + ", z = " + str(round(tvec[0, 0, 2], 2)), 
                        (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 顯示視窗
    cv2.imshow('Marker Detection and Pose Estimation', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 釋放資源
cv2.destroyAllWindows()
drone.streamoff()
