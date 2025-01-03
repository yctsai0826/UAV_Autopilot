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
target_distance = 0.4  # 米
max_speed_threshold = 20  # 最大速度限制
drone.send_rc_control(0,0,0,0)
zero = 0 

while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)
    
    frame = frame_read.frame
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 偵測 ArUco 標記
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

    # 畫出偵測到的標記
    if markerIds == 3:
        zero = 1 
        print(f'{markerIds=}')
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # 計算每個標記的姿態 (假設標記的邊長為 0.15 米)
        marker_length = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # 畫出坐標軸
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # 獲取 tvec 的 x, y, z 座標
            x, y, z = tvec[0]
            # z = z - 0.6     # meter
            # x = x 
            # y = y 
            # z = z 
            
            # 計算旋轉矩陣並將Z軸向量投影到XZ平面
            R, _ = cv2.Rodrigues(rvec)
            Z_axis = np.dot(R, np.array([0, 0, 1]))
            angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])  # Z 與 XZ 平面夾角

            # 用 math.degrees 轉換成度數
            yaw_degrees = math.degrees(angle_to_marker)
            yaw_degrees = yaw_degrees + 90

            # 構建顯示字串
            text = f"ID: {markerIds[i][0]}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m  yaw: {yaw_degrees:.2f} degrees"
            corner = markerCorners[i][0][0]
            cv2.putText(frame, text, (int(corner[0]), int(corner[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # 控制無人機移動，首先計算出需要的速度，並進行限制
            print(x, y, z, yaw_degrees)
            print(text)
            z_update = z_pid.update(50*(z - 0.7), 0)  # 控制無人機的前進和後退
            x_update = x_pid.update(50*x, 0)  # 控制無人機左右移動
            y_update = y_pid.update(50*y, 0)  # 控制無人機上下移動
            yaw_update = yaw_pid.update(yaw_degrees, 0)  # 控制無人機旋轉

            # 限制速度在合理範圍內
            z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
            x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
            y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)


            # 使用 send_rc_control 控制無人機的移動與旋轉
            drone.send_rc_control(int(1.5 * int(x_update)), int(z_update), 
                                  int(-2.5 * int(y_update)) , -1 * int(yaw_update))
    else:
        if(zero != 0):
            zero = 0
            drone.send_rc_control(0,0,0,0)
    # 顯示視窗
    cv2.imshow('Marker Detection and Pose Estimation', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 釋放資源
cv2.destroyAllWindows()
drone.streamoff()