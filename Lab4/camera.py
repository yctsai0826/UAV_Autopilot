import cv2 
import numpy as np 
import time 
import math 
from djitellopy import Tello 
from pyimagesearch.pid import PID 
 
CHECKERBOARD = (9, 6) 
 
def main(): 
         
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
 
 
    objpoints = []  # 3D  
    imgpoints = []  # 2D 
 
    # 3D point to [x,y, 0] 
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) 
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 
 
 
    drone=Tello() 
    drone.connect() 
    drone.streamon() 
    frame_read=drone.get_frame_read() 
 
    img_count = 0 
    while img_count < 100: 
        frame=frame_read.frame 
        cv2.waitKey(30) 
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        resized_frame = cv2.resize(frame, (640, 480)) 
 
         
 
        cv2.imshow('frame', resized_frame) 
        cv2.moveWindow('frame', 0, 0) 
 
        # detect corners 
        ret, corners = cv2.findChessboardCorners( 
            gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE 
        ) 
 
        if ret: 
            # prepare for subpixel accuracy 
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
             
             
            frame_with_corners = cv2.drawChessboardCorners(frame.copy(), CHECKERBOARD, corners, ret) 
            resized_frame2 = cv2.resize(frame_with_corners, (640, 480)) 
             
            cv2.imshow('frame2', resized_frame2) 
            cv2.moveWindow('frame2', 640,0)  
 
            # save 
            objpoints.append(objp) 
            imgpoints.append(corners) 
            cv2.imwrite(f"./imgs/image{img_count}.jpg", frame_with_corners) 
            img_count += 1 
 
         
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
 
 
    cv2.destroyAllWindows() 
 
 
    if len(objpoints) >= 4: 
        print("開始計算相機校準參數...") 
         
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
        # print(objpoints, imgpoints) 
        # 儲存參數到 XML 檔案 
        fs = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE) 
        fs.write("camera_matrix", mtx) 
        fs.write("dist_coeff", dist) 
        fs.release() 
 
        print("saved calibration.xml") 
 
 
 
 
if __name__ == '__main__': 
    main()