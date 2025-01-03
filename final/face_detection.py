#error message 1205 旋轉角度always 很大，調整send control 中的值
import cv2 
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from Tello_Video.keyboard_djitellopy import keyboard


# 從 calibration.xml 讀取相機標定參數
fs = cv2.FileStorage('Camera.xml', cv2.FILE_STORAGE_READ)
intrinsic = fs.getNode('camera_matrix').mat()
distortion = fs.getNode('dist_coeff').mat()
fs.release()


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
d = 0.4
max_speed_threshold = 20  # 最大速度限制
zero = 0

def face(drone,yraw_d = 10):
    frame_read = drone.get_frame_read()
    face_width_cm = 15  # 假設人臉寬度是 15 公分
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    objp_face = np.array([[0, 0, 0], [face_width_cm, 0, 0], [face_width_cm, face_width_cm, 0], [0, face_width_cm, 0]], dtype=np.float32)
    approach = False
    
    while True:
        frame = frame_read.frame
        # 偵測臉部
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(40, 40))
        if len(faces):
            detect1 = True

        if detect1:
            for (f_x, f_y, f_w, f_h) in faces:
                cv2.rectangle(frame, (int(f_x), int(f_y), (int(f_x) + int(f_w)), (int(f_y) + int(f_h))), (0, 255, 0), 2)
                
                # 假設已知人臉的寬度，將物體3D點和2D點對應
                obj_points = objp_face
                image_points = np.array([[int(f_x), int(f_y)], [int(f_x) + int(f_w), int(f_y)], [int(f_x) + int(f_w),int(f_y) + int(f_h)], [int(f_x), int(f_y) + int(f_h)]], dtype=np.float32)

                # 使用 solvePnP 計算旋轉向量和位移向量
                _, rvec, tvec = cv2.solvePnP(obj_points, image_points, intrinsic, distortion)
                tvec = tvec.flatten()  # 確保 tvec 是一維向量
                x, y, z = tvec #單位公尺
                x = x/100
                y= y/100
                z=z/100
                #print("x,y,z:",x,y,z)
                
                cv2.putText(frame, f"Distance: {z:.2f} cm", (int(f_x), int(f_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)           
                    
                # 計算旋轉矩陣並將Z軸向量投影到XZ平面
                R, _ = cv2.Rodrigues(rvec)
                Z_axis = np.dot(R, np.array([0, 0, 1]))
                angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])  # Z 與 XZ 平面夾角

                # 用 math.degrees 轉換成度數
                yaw_degrees = math.degrees(angle_to_marker)
                yaw_degrees -= 90
                #print(f"{yaw_degrees=}")

                if (x < 0.15 and x > -0.15 and y < 0.15 and y > -0.15 and z - d < 0.15 and z - d > -0.15 and yaw_degrees < yraw_d and yaw_degrees > (-1 * yraw_d)):
                    approach = True
                    break

                # 控制無人機移動
                z_update = z_pid.update(50 * (z - d), 0)  # 前進後退
                x_update = x_pid.update(50 * x, 0)  # 左右移動
                y_update = y_pid.update(50 * y, 0)  # 上下移動
                yaw_update = yaw_pid.update(yaw_degrees, 0)  # 旋轉

                # 限制速度範圍
                z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
                x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
                y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)
                yaw_update = np.clip(yaw_update, -max_speed_threshold, max_speed_threshold)

                #使用無人機的控制命令
                if (y_update > 0):
                    y_update = y_update * 1.5
                
                print("update: x,y,z",x_update,y_update,z_update,yaw_update)
                print(f"{yaw_update=}")
                drone.send_rc_control(int(1.5*x_update), int(z_update), int(-2.5 * int(y_update)), int(-10 * yaw_update))
                time.sleep(1)

        # 顯示結果
        cv2.imshow('Detection', frame)
        if (approach): 
            return True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':

    drone = Tello()
    drone.connect()
    drone.streamon()
    drone.send_rc_control(0, 0, 0, 0)
    while True: 
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        cv2.imshow('frame',frame)
        key = cv2.waitKey(50)  # 捕獲鍵盤按鍵
        if key != -1:
            keyboard(drone,key)
        if key == ord('7'):
            drone.send_rc_control(0, 40, 0, 0)#前
            time.sleep(1)
            drone.send_rc_control(0, 0, 0, 0)
            break
    try:
        face(drone) #detect face
        drone.send_rc_control(0, 0, 40, 0)#上
        time.sleep(3)
        drone.send_rc_control(0, 0, 0, 0)

        drone.send_rc_control(0, 40, 0, 0)#前
        time.sleep(1.5)
        drone.send_rc_control(0, 0, 0, 0)

        drone.send_rc_control(0, 0, -40, 0)#下
        time.sleep(5)
        drone.send_rc_control(0, 0, 0, 0)

        face(drone) #detect face
        drone.send_rc_control(0, 0, -40, 0)#下
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)

        drone.send_rc_control(0, 40, 0, 0)#前
        time.sleep(5)
        drone.send_rc_control(0, 0, 0, 0)

        drone.send_rc_control(-40, 0, 0, 0)#左
        time.sleep(1)
        drone.send_rc_control(0, 0, 0, 0)
        
    except Exception as e:
        print(f"出現錯誤: {e}")
    finally:
        drone.land()




