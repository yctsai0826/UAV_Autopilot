import cv2
import numpy as np

# 初始化 HOG 描述符並設定預設人員檢測器
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 載入 Haar-cascade 設定檔
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

fs = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeff').mat()

# 物理世界中的實際點 (以平面為例，單位: cm)
objp = np.array([
    [0.0, 0.0, 0.0],  # 左上角
    [0.0, 13.0, 0.0],  # 左下角
    [13.0, 0.0, 0.0],  # 右上角
    [13.0, 13.0, 0.0]  # 右下角
], dtype=np.float32)

import numpy as np

# 定義一個簡化的行人 3D 參考點集合
# 假設行人高度 170cm，寬度 50cm
objp2 = np.array([
    [0.0, 0.0, 0.0],      # 頭部左上角 (原點)
    [0.0, 1.7, 0.0],    # 腳部左下角
    [0.5, 0.0, 0.0],     # 頭部右上角
    [0.5, 1.7, 0.0]    # 腳部右下角
], dtype=np.float32)


# 開啟攝影機
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取畫面")
        break

    # 轉換為灰階影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 HOG 描述符檢測人員
    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05, padding=(32,32), useMeanshiftGrouping=False)
    for (x, y, w, h) in rects:
        # 假設使用人員框中的四角作為影像點
        img_points = np.array([
            [x, y],  # 左上角
            [x, y + h],  # 左下角
            [x + w, y],  # 右上角
            [x + w, y + h]  # 右下角
        ], dtype=np.float32)

        # 使用 solvePnP 計算深度
        retval, rvec, tvec = cv2.solvePnP(objp2, img_points, camera_matrix, dist_coeffs)
        if retval:
            distance = np.linalg.norm(tvec)  # 計算距離
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # 假設 distance 是以 cm 為單位的計算結果
            distance_m = distance  # 將距離從公分轉換為公尺
            cv2.putText(frame, f'{distance_m:.2f} m', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 使用 Haar-cascade 偵測臉部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        img_points = np.array([
            [x, y],  # 左上角
            [x, y + h],  # 左下角
            [x + w, y],  # 右上角
            [x + w, y + h]  # 右下角
        ], dtype=np.float32)

        # 使用 solvePnP 計算深度
        retval, rvec, tvec = cv2.solvePnP(objp, img_points, camera_matrix, dist_coeffs)
        if retval:
            distance = np.linalg.norm(tvec)  # 計算距離
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{distance:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 顯示影像
    cv2.imshow('Real-time Detection with Depth Estimation', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機並關閉視窗
cap.release()
cv2.destroyAllWindows()
