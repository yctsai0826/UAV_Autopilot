import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

# 從 calibration.xml 讀取相機標定參數
fs = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeff').mat()
fs.release()

# 設定 ArUco 標記字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# 初始化偵測器參數
parameters = cv2.aruco.DetectorParameters()

drone = Tello()
drone.connect()
#time.sleep(10)
drone.streamon()
frame_read = drone.get_frame_read()

while True:
    # 讀取攝像頭畫面
    # ret, frame = cap.read()
    frame = frame_read.frame


    # 轉換為灰階（可選）
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 偵測 ArUco 標記
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

    # 畫出偵測到的標記
    if markerIds is not None:
        print(f'{markerIds=}')
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # 計算每個標記的姿態 (假設標記的邊長為 0.05 米)
        marker_length = 0.15  # 標記的實際邊長（單位：米）
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

        # 畫出坐標軸並顯示位移向量
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # 畫出坐標軸
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # 獲取 tvec 的 x, y, z 座標
            x, y, z = tvec[0]

            # 構建顯示字串
            text = f"ID: {markerIds[i][0]}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m"

            # 在標記附近顯示位移向量
            corner = markerCorners[i][0][0]  # 獲取標記的左上角點
            cv2.putText(frame, text, (int(corner[0]), int(corner[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255) , 2, cv2.LINE_AA)

    # 顯示視窗
    cv2.imshow('Marker Detection and Pose Estimation', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源並關閉視窗
# cap.release()
cv2.destroyAllWindows()