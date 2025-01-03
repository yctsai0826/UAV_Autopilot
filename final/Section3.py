import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from Tello_Video.keyboard_djitellopy import keyboard
import numpy as np
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords

drone = Tello()
drone.connect()
drone.streamon()


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

fs = cv2.FileStorage('Camera.xml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeff').mat()
fs.release()

x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.4, kI=0.0, kD=0.2)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

max_speed_threshold = 20
marker_length = 0.15
threshold = 30
black_threshold = 0.15

d = 0.6
dh = 0
dw = 0
dx = dy = dz = 0.15     # error
yraw_d = 10

# last
device = "cuda" if torch.cuda.is_available() else "cpu"
model = attempt_load('best.pt', map_location=device)

def detect_drone_object_noshow(frame, model=None, device=None):
   """
   處理無人機的單一影像幀
   Args:
       frame: 無人機的影像幀
       model: 預先載入的模型(如果是 None 則會重新載入)
       device: 運算裝置(如果是 None 則會重新設定)
   Returns:
       檢測到的物體類別名稱，如果沒有檢測到則返回 None
   """
   # 如果沒有提供模型和裝置，進行初始化
   if model is None or device is None:
       device = "cuda" if torch.cuda.is_available() else "cpu"
       model = attempt_load('best.pt', map_location=device)
       if device == "cuda":
           model = model.half().to(device)
       else:
           model = model.float().to(device)
   
   # 獲取類別名稱
   names = model.module.names if hasattr(model, 'module') else model.names
   
   # 預處理圖片
   frame = letterbox(frame, (640, 640), stride=64, auto=True)[0]
   if device == "cuda":
       frame = transforms.ToTensor()(frame).to(device).half().unsqueeze(0)
   else:
       frame = transforms.ToTensor()(frame).to(device).float().unsqueeze(0)

   # 進行推理
   with torch.no_grad():
       pred = model(frame)[0]
   pred = non_max_suppression_kpt(pred, 0.25, 0.65)[0]

   # 檢查結果
   if pred is not None and len(pred) > 0:
       for *_, conf, cls in pred:
           class_id = int(cls)
           return names[class_id]  # 返回第一個檢測到的物體類別
   
   return None  # 如果沒有檢測到任何物體

def detect_drone_object(frame, model=None, device=None):
   """
   處理無人機的單一影像幀並顯示檢測結果
   Args:
       frame: 無人機的影像幀 
       model: 預先載入的模型
       device: 運算裝置
   Returns:
       檢測到的物體類別名稱，如果沒有檢測到則返回 None
   """
   # 儲存原始圖片用於顯示
   original_frame = frame.copy()
   
   # 如果沒有提供模型和裝置，進行初始化
   if model is None or device is None:
       device = "cuda" if torch.cuda.is_available() else "cpu"
       model = attempt_load('best.pt', map_location=device)
       if device == "cuda":
           model = model.half().to(device)
       else:
           model = model.float().to(device)
   
   # 獲取類別名稱
   names = model.module.names if hasattr(model, 'module') else model.names
   
   # 生成隨機顏色
   colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
   
   # 預處理圖片
   frame = letterbox(frame, (640, 640), stride=64, auto=True)[0]
   if device == "cuda":
       frame = transforms.ToTensor()(frame).to(device).half().unsqueeze(0)
   else:
       frame = transforms.ToTensor()(frame).to(device).float().unsqueeze(0)

   # 進行推理
   with torch.no_grad():
       pred = model(frame)[0]
   pred = non_max_suppression_kpt(pred, 0.25, 0.65)[0]

   detected_class = None
   # 檢查結果並繪製檢測框
   if pred is not None and len(pred) > 0:
       # 調整預測框的座標以匹配原始圖片大小
       pred[:, :4] = scale_coords(frame.shape[2:], pred[:, :4], original_frame.shape).round()
       
       for *xyxy, conf, cls in pred:
           class_id = int(cls)
           detected_class = names[class_id]
           
           # 繪製邊界框和標籤
           label = f'{names[class_id]} {conf:.2f}'
           color = colors[class_id]
           
           # 轉換座標為整數
           x1, y1, x2, y2 = map(int, xyxy)
           
           # 繪製邊界框
           cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
           
           # 添加標籤
           text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
           cv2.rectangle(original_frame, (x1, y1 - text_size[1] - 4), 
                        (x1 + text_size[0], y1), color, -1)
           cv2.putText(original_frame, label, (x1, y1 - 2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
   
   # 顯示結果
   cv2.imshow('Detection', original_frame)
   cv2.waitKey(1)
   
   return detected_class
def follow_marker(drone, markerID):
    frame_read = drone.get_frame_read()
    
    while True:
        frame = frame_read.frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)
        
        cv2.imshow('drone', frame)
        key = cv2.waitKey(33)
        if key != -1:
            break
        
        if markerIds is not None and markerID in markerIds:
            # Remove non-target markers
            markerIds = np.array(markerIds).flatten()
            valid_idx = np.where(markerIds == markerID)[0]
            markerIds = markerIds[valid_idx]
            markerCorners = [markerCorners[i] for i in valid_idx[0]]
            
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)
            
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                R, _ = cv2.Rodrigues(rvec)
                Z_axis = np.dot(R, np.array([0, 0, 1]))
                angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])
                
                yaw_degrees = math.degrees(angle_to_marker)
                yaw_degrees += 90
                
                x, y, z = tvec[0]
                if (x - dw < dx and x - dw > -1 * dx and y - dh < dy and y - dh > -1 * dy and z - d < dz and z - d > -1 * dz and yaw_degrees < yraw_d and yaw_degrees > (-1 * yraw_d)):
                    print("----------Marker detected-----------")
                    drone.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.1)
                    return
                drone.move()
                text = f"ID: {markerIds[0]}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m  yaw: {yaw_degrees:.2f} degrees"
                corner = markerCorners[i][0][0]
                cv2.putText(frame, text, (int(corner[0]), int(corner[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                
                z_update = z_pid.update(50 * (z - d), 0)
                x_update = x_pid.update(50 * x, 0)
                y_update = y_pid.update(50 * (y - dh), 0)
                yaw_update = yaw_pid.update(yaw_degrees, 0)
                if (y_update > 0):
                    y_update = y_update * 1.5
                z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
                x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
                y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)
                

                    
                drone.send_rc_control(int(1.5 * int(x_update)), int(1.5 * z_update),
                                      int(-2.5 * int(y_update)), -1 * int(yaw_update))
        else:
            drone.send_rc_control(0, 0, 0, 0)  # 如果沒有檢測到 marker，停止移動

def follow_marker_direction(drone, markerID, direction='left',target_distance = 0.5, distance_threshold = 0.1):
    """
    追蹤特定 marker 並在距離 50cm 時降落
    參數:
        drone: Tello 無人機對象
        markerID: 要追蹤的 marker ID
        direction: 'left' 或 'right'，決定追蹤左邊還是右邊的 marker
    """
    # 獲取無人機的視訊流

    
    while True:
        frame_read = drone.get_frame_read()
        # 獲取當前畫面並轉換為灰階
        frame = frame_read.frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 偵測 ArUco 標記
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)
        
        # 顯示畫面
        cv2.imshow('drone', frame)
        key = cv2.waitKey(33)
        if key != -1:  # 如果有按鍵輸入則退出
            break
        
        # 如果檢測到目標 marker
        if markerIds is not None and markerID in markerIds:
            # 將 markerIds 轉換為一維陣列
            markerIds = np.array(markerIds).flatten()
            print(f"markerIds = {markerIds}\n")
            # 找出目標 marker 的索引
            valid_idx = np.where(markerIds == markerID)[0]
            print(f"valid_idx = {valid_idx}\n")
            # 如果檢測到多個相同 ID 的 marker
            if len(valid_idx) > 1:
                # 計算每個 marker 的中心 x 座標
                center_xs = []
                for idx in valid_idx:
                    corners = markerCorners[idx][0]
                    center_x = np.mean(corners[:, 0])  # 計算四個角點的平均 x 座標
                    center_xs.append((idx, center_x))
                
                # 根據指定方向選擇目標 marker
                if direction == 'left':
                    # 選擇最左邊的 marker（x座標最小）
                    target_marker = min(center_xs, key=lambda x: x[1])
                else:
                    # 選擇最右邊的 marker（x座標最大）
                    target_marker = max(center_xs, key=lambda x: x[1])
                
                # 只保留選定的 marker
                valid_idx = [target_marker[0]]
                print(f"檢測到多個 marker，選擇{direction}側的一個，x座標: {target_marker[1]}")
            
            # 更新 marker 資訊
            markerIds = markerIds[valid_idx]
            markerCorners = [markerCorners[i] for i in valid_idx]
            
            # 在畫面上標示檢測到的 marker
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            # 計算 marker 的位姿
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)
            
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                # 繪製坐標軸
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                # 計算 marker 的朝向
                R, _ = cv2.Rodrigues(rvec)
                Z_axis = np.dot(R, np.array([0, 0, 1]))
                angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])
                yaw_degrees = math.degrees(angle_to_marker)
                yaw_degrees += 90
                
                # 獲取位置資訊
                x, y, z = tvec[0]
                # 檢查是否達到降落條件
                if (abs(z - target_distance) < distance_threshold and  # 前後距離在範圍內
                    abs(x) < 0.15 and                                # 左右位置接近中心
                    abs(y) < 0.15 and                                # 上下位置接近中心
                    abs(yaw_degrees) < 10):                          # 朝向正確
                    # print(f"目前距離")
                    print(f"達到目標距離: {z:.2f}m，準備降落")
                    drone.send_rc_control(0, 0, 0, 0)  # 停止移動
                    time.sleep(1)  # 等待穩定
                    drone.land()   # 執行降落
                    return
                
                # 在畫面上顯示位置和朝向資訊
                text = f"ID: {markerIds[0]}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m  yaw: {yaw_degrees:.2f} degrees"
                corner = markerCorners[i][0][0]
                cv2.putText(frame, text, (int(corner[0]), int(corner[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                
                # PID 控制計算更新值
                z_update = z_pid.update(50 * (z - target_distance), 0)  # 前後移動
                x_update = x_pid.update(50 * x, 0)                      # 左右移動
                y_update = y_pid.update(50 * y, 0)                      # 上下移動
                yaw_update = yaw_pid.update(yaw_degrees, 0)             # 旋轉角度
                
                                # 上升時增加速度補償
                if (y_update > 0):
                    y_update = y_update * 1.5
                # 限制移動速度在安全範圍內
                z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
                x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
                y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)
                

                
                # 發送控制命令給無人機
                drone.send_rc_control(int(1.5 * int(x_update)), int(1.5 * z_update),
                                      int(-2.5 * int(y_update)), -1 * int(yaw_update))

def follow_line(drone, direction, target_corner, horizontal):
    cur_squares = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    while not compare_squares(target_corner, cur_squares):
        frame = drone.get_frame_read().frame
        process_frame = img_process(frame)
        
        cur_squares, ratio, is_black = line_shape(process_frame)
        
        frame_black = put_detected_square(frame, cur_squares, ratio)
        
        cv2.imshow('drone_black', frame_black)
        cv2.imshow('drone', frame)
        print(cur_squares)
        
        key = cv2.waitKey(50)
        if key != -1:
            keyboard(drone, key)
            
        if is_black:
            drone.send_rc_control(0, -5, 0, 0)  # Too close to the black line
        else:
            if horizontal and cur_squares[0:3] == [1, 1, 1]:
                drone.send_rc_control(0, 0, 5, 0)
            elif horizontal and cur_squares[6:9] == [1, 1, 1]:
                drone.send_rc_control(0, 0, -5, 0)
            elif not horizontal and cur_squares[::3] == [1, 1, 1]:
                drone.send_rc_control(-5, 0, 0, 0)
            elif not horizontal and cur_squares[2::3] == [1, 1, 1]:
                drone.send_rc_control(5, 0, 0, 0)
            else:
                drone.send_rc_control(direction[0], direction[1], direction[2], direction[3])
            
    drone.send_rc_control(0, 0, 0, 0)
   
    
def compare_squares(targets, cur):
    # for i in range(9):
    #     if target[i] == 2:
    #         cur[i] = 2
    for target in targets:
        if(target == cur):
            return True
    return False
    
    
def img_process(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=0.2, fy=0.2)
    Blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thres = cv2.threshold(Blur, threshold, 255, cv2.THRESH_BINARY)
    return thres
    

def line_shape(frame):
    h, w = frame.shape[:2]
    
    h_bound = [0, h//3, 2*h//3, h]
    w_bound = [0, w//3, 2*w//3, w]
        
    squares = {
        'tl': 0, 'tm': 0, 'tr': 0,
        'ml': 0, 'mm': 0, 'mr': 0,
        'bl': 0, 'bm': 0, 'br': 0
    }
    
    ratio = {
        'tl': 0, 'tm': 0, 'tr': 0,
        'ml': 0, 'mm': 0, 'mr': 0,
        'bl': 0, 'bm': 0, 'br': 0
    }
    
    total_pixels = (h // 3) * (w // 3)
    black_ratio_threshold = 0.1
    is_black = False
    for i, (h1, h2) in  enumerate(zip(h_bound[0:3], h_bound[1:4])):
        for j, (w1, w2) in enumerate(zip(w_bound[0:3], w_bound[1:4])):
            region = frame[h1:h2, w1:w2]
            black_pixels = np.sum(region == 0)
            black_ratio = black_pixels / total_pixels
            
            if (black_ratio > 0.7):
                is_black = True
                        
            pos = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br'][i*3 + j]
            squares[pos] = 1 if black_ratio > black_threshold else 0
            ratio[pos] = round(black_ratio, 2)
            
    return [squares['tl'], squares['tm'], squares['tr'], 
            squares['ml'], squares['mm'], squares['mr'], 
            squares['bl'], squares['bm'], squares['br']],[ratio['tl'], ratio['tm'], ratio['tr'], 
            ratio['ml'], ratio['mm'], ratio['mr'], 
            ratio['bl'], ratio['bm'], ratio['br']],is_black   


def put_detected_square(frame, cur_squares, ratio):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, Thres = cv2.threshold(Blur, threshold, 255, cv2.THRESH_BINARY)
    h, w = Thres.shape
    frame = Thres
    
    w_center = w // 2
    h_center = h // 2
    
    # 定義九宮格中心位置
    x_list = [30, w_center - w // 6 + 30, w - w // 3 + 30]  # 調整位置
    y_list = [30, h_center - h // 6 + 30, h - h // 3 + 30]  # 調整位置
    
    for i, (black, black_rate) in enumerate(zip(cur_squares, ratio)):
        x = x_list[i % 3]  # 水平位置
        y = y_list[i // 3]  # 垂直位置
        if black == 1:
            # 黑色文字顏色改成紅色
            cv2.putText(frame, text=f'black, {black_rate}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.7, org=(x, y), color=(0, 0, 255), thickness=1)  # 紅色 (BGR: 0, 0, 255)
        else:
            # 白色文字顏色改成藍色
            cv2.putText(frame, text=f'white, {black_rate}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.7, org=(x, y), color=(255, 0, 0), thickness=1)  # 藍色 (BGR: 255, 0, 0)
    
    frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    return frame


drone.send_rc_control(0, 0, 0, 0)

frame_read = drone.get_frame_read()
while True:
    frame = frame_read.frame
    cv2.imshow('drone', frame)
    key = cv2.waitKey(50)
    if key != -1:
        keyboard(drone, key)
    if key == ord('7'):
        print("start\n")
        break


# follow_line(drone, [-15, 0, 0, 0], [[0, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 1, 1]], True)
# print("Stage 1 finished")
# follow_line(drone, [0, 0, 15, 0], [[0, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0]], False)
# print("Stage 2 finished")
# follow_line(drone, [-15, 0, 0, 0], [[0, 0, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0]], True)
# print("Stage 3 finished")
# follow_line(drone, [0, 0, -15, 0], [[0, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 1, 1]], False)
# print("Stage 4 finished")

# last
follow_marker(drone, 2)
drone.rotate_clockwise(180)

while True:
    frame = frame_read.frame
    wawa = detect_drone_object(frame)
    if wawa =='Kanahera':
        direction = "left"
        break
    elif wawa == 'Melody':
        direction = "right"
        break
follow_marker_direction(drone, 3, direction)

drone.land()