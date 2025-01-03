import cv2
import numpy as np
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard  # 匯入鍵盤控制功能

def display_text_bottom_right(frame, text):
    """
    在畫面右下角顯示文字
    """
    frame_height, frame_width = frame.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 2)
    x_pos = frame_width - text_width - 300  # 調整邊界
    y_pos = frame_height - text_height - 100
    cv2.rectangle(frame, (x_pos, y_pos - text_height - 10), 
                  (x_pos + text_width, y_pos), (0, 0, 0), -1)
    cv2.putText(frame, text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    # 從 calibration.xml 讀取相機標定參數
    fs = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('dist_coeff').mat()
    fs.release()

    # 設定 ArUco 標記字典
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    # 啟動無人機
    drone = Tello()
    drone.connect()
    drone.streamon()  # 啟動無人機影像串流
    frame_read = drone.get_frame_read()
    # # 控制影像大小
    # frame = cv2.resize(frame, (480, 360))
    # 設定 PID 控制參數
    pid_x = PID(kP=0.8, kI=0.001, kD=0.1)  # 調整左右移動的 PID
    pid_y = PID(kP=0.8, kI=0.01, kD=0.12)   # 調整前後移動的 PID
    pid_z = PID(kP=0.8, kI=0.001, kD=0.1)  # 調整高度控制的 PID
    pid_yaw = PID(kP=0.8, kI=0.001, kD=0.1) # 調整旋轉控制的 PID

    pid_x.initialize()
    pid_y.initialize()
    pid_z.initialize()
    pid_yaw.initialize()

    # 目標參數
    
    target_z = 1.0  # 目標距離 (米)
    target_yaw = 0  # 目標旋轉角度
    target_x = 0    # 左右位置
    target_y = 0    # 前後位置

    auto_pilot = False  # 自動駕駛模式標誌
    loft = False        # 高度控制模式標誌
    
    while True:
        # 讀取無人機攝影畫面
        frame = frame_read.frame  # 獲取無人機的實時畫面
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測 ArUco 標記
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

        # 如果自動駕駛模式啟動且偵測到標記
        if auto_pilot and markerIds is not None:
            # 畫出偵測到的標記
            # frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            # 標記大小與座標估計
            marker_length = 0.15  # 標記的實際邊長
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)

            # 使用第一個標記進行控制
            rvec = rvecs[0]
            tvec = tvecs[0]
            marker_id = markerIds[0][0]

            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # 獲取 tvec 的 x, y, z 座標
            x, y, z = tvec[0]

            # 顯示標記ID與座標
            text = f"ID: {marker_id}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m"
            display_text_bottom_right(frame, text)
            
            # 旋轉矩陣計算偏航角
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) if sy > 1e-6 else math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw_degrees = math.degrees(yaw)

            # 使用 PID 控制器更新速度
            speed_yaw = pid_yaw.update(yaw_degrees, target_yaw) * 2   
            speed_x = pid_x.update(x, target_x) * 1                
            speed_y = pid_y.update(z, target_z) * 8         
            speed_z = pid_z.update(-y, target_y) * 1             

            
            
            # 限制速度
            speed_x = np.clip(speed_x, -100, 100)
            speed_y = np.clip(speed_y, -100, 100)
            speed_z = np.clip(speed_z, -100, 100)
            speed_yaw = np.clip(speed_yaw, -100, 100)

            speed_x = int(speed_x)
            speed_y = int(speed_y)
            speed_z = int(speed_z)
            speed_yaw = int(speed_yaw)
            
            # if z > target_z:  # 無人機距離標記太近，應後退
            #     speed_y = -abs(speed_y)  # 確保 speed_y 是負數
            # else:  # 無人機距離標記太遠，應前進
            #     speed_y = abs(speed_y)  # 確保 speed_y 是正數
            if(z < target_z):
                speed_y = -speed_y
            
            
            print(f"speed_x: {speed_x:.2f}, speed_y: {speed_y:.2f}, speed_z: {speed_z:.2f}, speed_yaw: {speed_yaw:.2f}")

            # 控制無人機移動
            drone.send_rc_control(speed_x, speed_y, speed_z, speed_yaw)
        elif auto_pilot and markerIds is None and not loft:
            loft = True
            drone.send_rc_control(0, 0, 0, 0)
        
        # 顯示無人機實時畫面
        cv2.imshow('Drone Camera Feed', frame)

        # 使用從 keyboard_djitellopy.py 匯入的鍵盤控制功能
        key = cv2.waitKey(1) & 0xFF
        
        # 切換自動駕駛模式
        if key == ord('r'):
            auto_pilot = not auto_pilot
            print(f"自動駕駛模式 {'開啟' if auto_pilot else '關閉'}")

        # 當自動駕駛關閉時，啟用手動控制
        if not auto_pilot:
            keyboard(drone, key)  # 調用鍵盤控制函數

        # 當按下 'q' 鍵時退出並降落無人機
        if key == ord('q'):
            drone.land()
            break
        

    # 關閉視窗並釋放資源
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
