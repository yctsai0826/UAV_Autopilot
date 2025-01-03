import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from Tello_Video.keyboard_djitellopy import keyboard

import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box


drone = Tello()
drone.connect()
drone.streamon()


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

WEIGHT = './best.pt'
device = "cpu"
model = attempt_load(WEIGHT, map_location=device).float().to(device)

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

# PID 控制器設置 (用於控制無人機的旋轉和前後移動)
x_facepid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_facepid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_facepid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_facepid = PID(kP=0.4, kI=0.0, kD=0.2)  # 控制無人機的旋轉

x_facepid.initialize()
y_facepid.initialize()
z_facepid.initialize()
yaw_facepid.initialize()

face_distance = 0.4

max_speed_threshold = 15
marker_length = 0.15
threshold = 30
black_threshold = 0.1

dh = 0
dw = 0
dx = dy = dz = 0.15
yraw_d = 5

finish = False


rt = [[0, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1, 0, 0, 1]]
lt = [[0, 0, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0]]
rb = [[0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1, 1]]
lb = [[0, 1, 0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 1, 1]]
corner = [[2, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 1, 1]]


def follow_marker(drone, markerID, distance, choose = 0, angle = 90):
    frame_read = drone.get_frame_read()
    
    while True:
        frame = frame_read.frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)
        
        key = cv2.waitKey(33)
        if key == ord('0'):
            print("----------------Exit--------------------")
            return True
        if key != -1:
            keyboard(drone, key)
        
        if markerIds is not None and markerID in markerIds:
            markerIds = np.array(markerIds).flatten()
            
            if len(markerCorners) >= 2 and choose != 0:
                valid_idxs = np.where(markerIds == markerID)[0]
                markerCorners = [markerCorners[i] for i in valid_idxs]
                centers = []
                for corner in markerCorners:
                    center_x = (corner[0][0][0] + corner[0][2][0]) / 2  # 左上角X和右下角X的平均值
                    centers.append(center_x)
                
                if choose == 1: # choose left
                    chosen_idx = np.argmin(centers)
                elif choose == 2: # choose right
                    chosen_idx = np.argmax(centers)
                
                markerCorners = [markerCorners[chosen_idx]]
                markerIds = np.array([markerIds[chosen_idx]])
            
            if choose == 0:
                valid_idx = np.where(markerIds == markerID)[0]
                markerCorners = [markerCorners[i] for i in valid_idx]
                markerIds = markerIds[valid_idx]
            
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_length, camera_matrix, dist_coeffs)
            
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                R, _ = cv2.Rodrigues(rvec)
                Z_axis = np.dot(R, np.array([0, 0, 1]))
                angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])
                
                yaw_degrees = math.degrees(angle_to_marker)
                yaw_degrees += angle
                
                x, y, z = tvec[0]
                if (x - dw < dx and x - dw > -1 * dx and y - dh < dy and y - dh > -1 * dy and z - distance < dz and z - distance > -1 * dz and yaw_degrees < yraw_d and yaw_degrees > (-1 * yraw_d)):
                    print(f"----------Marker {markerID} detected-----------")
                    drone.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.1)
                    return True
                
                text = f"ID: {markerIds[0]}  x: {x:.2f} m  y: {y:.2f} m  z: {z:.2f} m  yaw: {yaw_degrees:.2f} degrees"
                print(f"{yaw_degrees=}")
                corner = markerCorners[i][0][0]
                cv2.putText(frame, text, (int(corner[0]), int(corner[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                
                z_update = z_pid.update(50 * (z - distance), 0)
                x_update = x_pid.update(50 * x, 0)
                y_update = y_pid.update(50 * (y - dh), 0)
                yaw_update = yaw_pid.update(yaw_degrees, 0)
                
                z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
                x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
                y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)
                yaw_update *= 3
                yaw_update = np.clip(yaw_update, -max_speed_threshold, max_speed_threshold)
                
                # if (y_update > 0):
                #     y_update = y_update * 1.5
                    
                drone.send_rc_control(int(2 * int(x_update)), int(1.5 * z_update),
                                      int(-2.5 * int(y_update)), -1 * int(yaw_update))
        else:
            drone.send_rc_control(0, 0, 0, 0)
            
        cv2.imshow('drone', frame)
     
       
def follow_line(drone, direction, target_corner, horizontal, adjustup = True, back_threshold = 0.6, stage = -1):

    cur_squares = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    while not compare_squares(target_corner, cur_squares):
        frame = drone.get_frame_read().frame
        process_frame = img_process(frame)
        
        cur_squares, ratio, is_black = line_shape(process_frame, back_threshold=back_threshold)
        
        frame_black = put_detected_square(frame, cur_squares, ratio)
        
        cv2.imshow('drone_black', frame_black)
        cv2.imshow('drone', frame)
        print(cur_squares)
        print(f"--------------------------{stage=}----------------------------")
        key = cv2.waitKey(50)
        if key == ord('m'):
            return True
        elif key != -1:
            keyboard(drone, key)
            
        if is_black:
            drone.send_rc_control(0, -8, 0, 0)  # Too close to the black line
        else:
            if adjustup:
                if horizontal and cur_squares[0:3] == [1, 1, 1]:
                    drone.send_rc_control(0, 0, 10, 0)
                elif horizontal and cur_squares[6:9] == [1, 1, 1]:
                    drone.send_rc_control(0, 0, -10, 0)
                elif not horizontal and cur_squares[::3] == [1, 1, 1]:
                    drone.send_rc_control(-10, 0, 0, 0)
                elif not horizontal and cur_squares[2::3] == [1, 1, 1]:
                    drone.send_rc_control(10, 0, 0, 0)
                else:
                    drone.send_rc_control(direction[0], direction[1], direction[2], direction[3])
                time.sleep(0.2)
            if not adjustup:
                pass
            
    drone.send_rc_control(0, 0, 0, 0)
    # time.sleep(1)
    return True
   
    
def compare_squares(targets, cur):
    tmp = cur
    for target in targets:
        for i in range(9):
            if target[i] == 2:
                cur[i] = 2
        if(target == cur):
            return True
        cur = tmp
    return False
    
    
def img_process(frame, scale=0.2):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
    Blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thres = cv2.threshold(Blur, threshold, 255, cv2.THRESH_BINARY)
    return thres
    

def line_shape(frame, back_threshold):
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
            
            if (black_ratio > back_threshold):
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
    frame = img_process(frame, 0.4)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # show text color
    h, w = frame.shape[:2]
    
    w_center = w // 2
    h_center = h // 2
    
    x_list = [10, w_center - w // 6 + 10, w - w // 3 + 10]
    y_list = [10, h_center - h // 6 + 10, h - h // 3 + 10]
    
    for i, (black, black_rate) in enumerate(zip(cur_squares, ratio)):
        x = x_list[i % 3]
        y = y_list[i // 3]
        if black == 1:
            cv2.putText(frame, text=f'black, {black_rate}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.7, org=(x, y), color=(0, 0, 255), thickness=1)
        else:
            cv2.putText(frame, text=f'white, {black_rate}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.7, org=(x, y), color=(255, 0, 0), thickness=1)
    
    return frame


def face(drone, yraw_d = 10):
    frame_read = drone.get_frame_read()
    face_width_cm = 15  # 假設人臉寬度是 15 公分
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    objp_face = np.array([[0, 0, 0], [face_width_cm, 0, 0], [face_width_cm, face_width_cm, 0], [0, face_width_cm, 0]], dtype=np.float32)
    approach = False
    
    # 加入穩定性判斷
    stable_count = 0
    last_face = None
    required_stable_frames = 3
    
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
            
        # 偵測臉部 - 增加參數限制
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,  # 提高至8，減少誤檢
            minSize=(80, 80),  # 提高最小尺寸
            maxSize=(300, 300)  # 限制最大尺寸
        )
        
        # 找到最大的臉
        if len(faces) > 0:
            max_area = 0
            max_face = None
            for face_rect in faces:
                area = face_rect[2] * face_rect[3]
                if area > max_area:
                    max_area = area
                    max_face = face_rect
            
            if max_face is not None:
                f_x, f_y, f_w, f_h = max_face
                
                # 確認臉部位置的穩定性
                if last_face is not None:
                    dx = abs(f_x - last_face[0])
                    dy = abs(f_y - last_face[1])
                    dw = abs(f_w - last_face[2])
                    dh = abs(f_h - last_face[3])
                    
                    if dx < 20 and dy < 20 and dw < 20 and dh < 20:
                        stable_count += 1
                    else:
                        stable_count = 0
                
                last_face = (f_x, f_y, f_w, f_h)
                
                # 只有當位置穩定時才進行追蹤
                if stable_count >= required_stable_frames:
                    # 繪製人臉框
                    cv2.rectangle(frame, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 0), 2)
                    
                    try:
                        # 3D 點和 2D 點對應
                        obj_points = objp_face
                        image_points = np.array([
                            [f_x, f_y],
                            [f_x + f_w, f_y],
                            [f_x + f_w, f_y + f_h],
                            [f_x, f_y + f_h]
                        ], dtype=np.float32)

                        # 計算位置和角度
                        _, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
                        tvec = tvec.flatten()
                        x, y, z = tvec
                        x = x / 100
                        y = y / 100
                        z = z / 100
                        
                        R, _ = cv2.Rodrigues(rvec)
                        Z_axis = np.dot(R, np.array([0, 0, 1]))
                        angle_to_marker = math.atan2(Z_axis[2], Z_axis[0])
                        yaw_degrees = math.degrees(angle_to_marker) - 90
                        
                        # 顯示資訊
                        cv2.putText(frame, f"Distance: {z:.2f} m", (f_x, f_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"X: {x:.2f} Y: {y:.2f} Z: {z:.2f}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, f"Yaw: {yaw_degrees:.2f}",
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # 檢查是否達到目標位置¡
                        if (abs(x) < 0.15 and abs(y) < 0.15 and
                            abs(z - face_distance) < 0.15 and
                            abs(yaw_degrees) < yraw_d):
                            drone.send_rc_control(0, 0, 0, 0)
                            approach = True
                            break

                        # PID 控制
                        z_update = z_facepid.update(50 * (z - face_distance), 0)
                        x_update = x_facepid.update(50 * x, 0)
                        y_update = y_facepid.update(50 * y, 0)
                        yaw_update = yaw_facepid.update(yaw_degrees, 0)

                        # 限制速度範圍
                        z_update = np.clip(z_update, -max_speed_threshold, max_speed_threshold)
                        x_update = np.clip(x_update, -max_speed_threshold, max_speed_threshold)
                        y_update = np.clip(y_update, -max_speed_threshold, max_speed_threshold)
                        yaw_update = np.clip(yaw_update, -max_speed_threshold, max_speed_threshold)
                        if x_update > 0:
                            x_update *= 2

                        # 控制無人機
                        drone.send_rc_control(int(1.5 * int(x_update)), int(1.5 * z_update),
                                      int(-2.5 * int(y_update)), -1 * int(yaw_update))
                    except Exception as e:
                        print(f"Error in face tracking: {e}")
                        continue
        else:
            stable_count = 0
            last_face = None
            drone.send_rc_control(0, 0, 0, 0)
        
        # 顯示畫面
        cv2.imshow('Detection', frame)
        key = cv2.waitKey(1)
        
        if approach:
            cv2.destroyAllWindows()
            return True
        
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False

        time.sleep(0.05)


def detect_doll(drone):
    print("test")
    frame_read = drone.get_frame_read()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    while True:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            print("Invalid frame. Retrying...")
            time.sleep(0.1)  # 適當等待
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_orig = frame.copy()
        frame = letterbox(frame, (640, 640), stride=64, auto=False)[0]
        frame = transforms.ToTensor()(frame).to(device).float().unsqueeze(0)

        if frame.shape[2:] != (640, 640):
            print(f"Invalid input size: {frame.shape[2:]}")
            continue

        try:
            with torch.no_grad():
                output = model(frame)[0]
            output = non_max_suppression_kpt(output, 0.25, 0.65)[0]
        except Exception as e:
            print(f"YOLO inference error: {e}")
            continue

        if len(output) == 0:
            continue

        output[:, :4] = scale_coords(frame.shape[2:], output[:, :4], frame_orig.shape).round()

        for *xyxy, conf, cls in output:
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            label = f"{names[int(cls)]} {conf:.2f}"

            color = colors[int(cls)]
            cv2.rectangle(frame_orig, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame_orig, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            print(f"Doll center: ({center_x}, {center_y}), Box area: {box_area}")

            print(names[int(cls)])
            if names[int(cls)] == "Kanahera" and conf > 0.6:
                cv2.imshow("Drone View", frame_orig)  # 顯示繪製後的影像
                return 1
            elif names[int(cls)] == "Melody" and conf > 0.6:
                cv2.imshow("Drone View", frame_orig)  # 顯示繪製後的影像
                return 2

        cv2.imshow("Drone View", frame_orig)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def Section_1(drone):
    try:
        face(drone) #detect face
        time.sleep(1)
        drone.send_rc_control(0, 0, 40, 0)#上
        time.sleep(3)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        
        drone.send_rc_control(0, 40, 0, 0)#前
        time.sleep(1.3)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)

        drone.send_rc_control(0, 0, -40, 0)#下
        time.sleep(6.5)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        
        drone.send_rc_control(15, 0, 0, 0)#下
        time.sleep(1)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)

        face(drone) #detect face
        time.sleep(1)
        drone.send_rc_control(0, 0, -40, 0)#下
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)

        drone.send_rc_control(0, 40, 0, 0)#前
        time.sleep(5)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)

        # drone.send_rc_control(-40, 0, 0, 0)#左
        # time.sleep(1)
        # drone.send_rc_control(0, 0, 0, 0)
        
        drone.send_rc_control(0, 0, 20, 0)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        
    except Exception as e:
        print(f"出現錯誤: {e}")
        

def Section_2(drone, doll):
    if doll == 1:
        while not follow_line(drone, [0, 0, 15, 0], rt, False):
            pass
        print("\n------------Stage 1 finish------------\n")
        while not follow_line(drone, [-15, 0, 0, 0],  [[0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0, 0], [0, 2, 2, 1, 1, 1, 1, 0, 0]], True):
            pass
        print("\n------------Stage 2 finish------------\n")
        while not follow_line(drone, [0, 0, -15, 0], [[0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0]], False):
            pass
        print("\n------------Stage 3 finish------------\n")
        while not follow_line(drone, [-15, 0, 0, 0], lb, True):
            pass
        # drone.send_rc_control(-40, 0, 0, 0)
        # time.sleep(3)
        # drone.send_rc_control(0, 0, 0, 0)
        # time.sleep(0.5)
        # drone.send_rc_control(0, 0, 20, 0)
        # time.sleep(2)
        print("\n------------Stage 4 finish------------\n")
        while not follow_line(drone, [0, 0, 15, 0], rt, False):
            pass
        print("\n------------Stage 5 finish------------\n")
        while not follow_line(drone, [-15, 0, 0, 0], corner, True):
            pass
        print("\n------------Stage 6 finish------------\n")
        while not follow_line(drone, [0, 0, 15, 0], rt, False):
            pass
        print("\n------------Stage 7 finish------------\n")
        while not follow_line(drone, [-15, 0, 0, 0], lt, True):
            pass
        print("\n------------Stage 8 finish------------\n")
        while not follow_line(drone, [0, 0, -15, 0], corner, False):
            pass
        print("\n------------Stage 9 finish------------\n")
        time.sleep(0.5)
        drone.send_rc_control(-20, 0, 0, 0)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        drone.send_rc_control(0, -20, 0, 0)
        time.sleep(1.5)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        
    elif doll == 2:
        while not follow_line(drone, [0, 0, 15, 0], rt, False, 1):
            pass
        print("\n------------Stage 1 finish------------\n")
        while not follow_line(drone, [-10, 0, 0, 0], corner, True, 2):
            pass
        print("\n------------Stage 2 finish------------\n")
        while not follow_line(drone, [0, 0, 15, 0], rt, False, 3):
            pass
        print("\n------------Stage 3 finish------------\n")
        while not follow_line(drone, [-10, 0, 0, 0], lt, True, 4):
            pass
        print("\n------------Stage 4 finish------------\n")
        while not follow_line(drone, [0, 0, -10, 0], corner, False, 5):
            pass
        print("\n------------Stage 5 finish------------\n")
        while not follow_line(drone, [-10, 0, 0, 0],  [[0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0, 0], [0, 2, 2, 1, 1, 1, 1, 0, 0]], True, 6):
            pass
        print("\n------------Stage 6 finish------------\n")
        while not follow_line(drone, [0, 0, -15, 0], [[0, 1, 0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0]], False, 7):
            pass
        print("\n------------Stage 7 finish------------\n")
        while not follow_line(drone, [-15, 0, 0, 0], lb, True, 8):
            pass
        time.sleep(1)
        # drone.send_rc_control(-40, 0, 0, 0)
        # time.sleep(3)
        # drone.send_rc_control(0, 0, 0, 0)
        # time.sleep(0.5)
        # drone.send_rc_control(0, 0, 20, 0)
        # time.sleep(2)
        print("\n------------Stage 8 finish------------\n")
        while not follow_line(drone, [0, 0, 15, 0], rt, False, 9):
            pass
        print("\n------------Stage 9 finish------------\n")
        drone.send_rc_control(-40, 0, 0, 0)
        time.sleep(3)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        drone.send_rc_control(0, -20, 0, 0)
        time.sleep(2.5)
        # drone.send_rc_control(0, 0, 0, 0)
        # time.sleep(0.5)
        # drone.send_rc_control(-40, 0, 0, 0)
        # time.sleep(2.5)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)






# Start Section

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
        drone.send_rc_control(0, 30, 0, 0)
        time.sleep(1)
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        break
    
    
#--------------------------------------Section 1--------------------------------------------#


# Section 1

# Section_1(drone)

#--------------------------------------Section 2--------------------------------------------#

# Section 2 - detect doll

doll = detect_doll(drone)
print(f"Find doll 1: {doll=}")

# Section 2 - detect marker

drone.send_rc_control(0, 0, 25, 0)
time.sleep(3.5)
drone.send_rc_control(0, 0, 0, 0)
time.sleep(0.5)

print("\nStart detect marker 1\n")
while not follow_marker(drone, 1, distance = 0.5, choose=0, angle=80):
    pass
print("\nFollow marker 1 finish\n")

# Section 2 - Follow line

Section_2(drone, doll)

#--------------------------------------Section 3--------------------------------------------#

# Section 3 - detect marker

drone.send_rc_control(0, 0, 0, 0)
time.sleep(1)

yraw_d = 20

print("\nStart detect marker 2\n")
while not follow_marker(drone, 2, distance = 0.7):
    pass
print("\nFollow marker 2 finish\n")
drone.rotate_clockwise(180)

# Section 3 - detect doll

max_speed_threshold = 8

doll = detect_doll(drone)
print(f"Find doll 2: {doll=}")

drone.send_rc_control(0, 0, 15, 0)
time.sleep(1)

# Section 3 - Follow marker
print("\nStart detect marker 3\n")
if doll == 1:
    while not follow_marker(drone, 3, distance = 0.5, choose=1):
        pass
    drone.send_rc_control(0, 0, 0, 0)
    print("\nFollow marker 3 finish\n")
elif doll == 2:
    while not follow_marker(drone, 3, distance = 0.5, choose=2):
        pass
    drone.send_rc_control(0, 0, 0, 0)
    print("\nFollow marker 3 finish\n")
    
drone.land()

