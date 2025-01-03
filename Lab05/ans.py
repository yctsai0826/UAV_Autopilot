import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

# 限制速度的函數
def clamp(x, max_speed_threshold=50):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x

# 鍵盤控制函數
def keyboard(self, key):
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30

    if key == ord('1'):
        self.takeoff()
    if key == ord('2'):
        self.land()
    if key == ord('3'):
        self.send_rc_control(0, 0, 0, 0)
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
    if key == ord('s'):
        self.send_rc_control(0, -fb_speed, 0, 0)
    if key == ord('a'):
        self.send_rc_control(-lf_speed, 0, 0, 0)
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
    if key == ord('x'):
        self.send_rc_control(0, 0, -ud_speed, 0)
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, -degree)
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print(battery)

# 無人機初始化
drone = Tello()
drone.connect()
drone.streamon()

# 讀取相機校正參數
f = cv2.FileStorage("Camera.xml", cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("camera_matrix").mat()
distortion = f.getNode("dist_coeff").mat()
f.release()

# 初始化 PID 控制器
x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

# 設定 ArUco 標記字典
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    frame = drone.get_frame_read().frame
    cv2.imshow("Drone Feed", frame)
    key = cv2.waitKey(1)

    # 偵測 ArUco 標記
    markerCorners, markerids, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    if markerids is not None:
        # 計算標記姿態
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

        for i in range(rvec.shape[0]):
            id = markerids[i][0]
            if id != 0:
                continue
            rotM = np.zeros(shape=(3, 3))
            cv2.Rodrigues(rvec[i], rotM)
            ypr = cv2.RQDecomp3x3(rotM)[0]
            yaw_update = ypr[1] * 1.2

            # 修改索引，移除第三個索引，直接訪問 tvec 的兩個維度
            x_update = tvec[i][0] - 10  # `tvec[i][0]` 是第 `i` 個標記的 x 坐標
            y_update = -(tvec[i][1] - (-20))  # `tvec[i][1]` 是第 `i` 個標記的 y 坐標
            z_update = tvec[i][2] - 150  # `tvec[i][2]` 是第 `i` 個標記的 z 坐標

            x_update = clamp(x_pid.update(x_update, sleep=0))
            y_update = clamp(y_pid.update(y_update, sleep=0))
            z_update = clamp(z_pid.update(z_update, sleep=0))
            yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))

            print(x_update, y_update, z_update, yaw_update)

            drone.send_rc_control(0, int(z_update // 2), int(y_update), int(yaw_update))

            # 繪製標記和坐標軸
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerids)
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[i], tvec[i], 10)

            # 顯示位置信息
            cv2.putText(frame, f"x = {round(tvec[i][0], 2)}, y = {round(tvec[i][1], 2)}, z = {round(tvec[i][2], 2)}", 
                        (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow("Drone Feed", frame)
        key = cv2.waitKey(1)
    else:
        drone.send_rc_control(0, 0, 0, 0)  # 如果沒有標記，無人機停止移動

    if key != -1:
        keyboard(drone, key)

    # 按 'q' 鍵退出
    if key == ord('q'):
        break

# 釋放資源
cv2.destroyAllWindows()
drone.streamoff()
