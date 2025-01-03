import cv2
import numpy as np

# Step 1: Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
patternSize = (8, 5)  # Columns, Rows
objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Step 2: Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 3: Find the chessboard corners
    retc, corners = cv2.findChessboardCorners(gray, patternSize, None)
    if retc == True:
        # Step 3: Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, patternSize, corners2, retc)
        # Store the image points and object points
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"Chessboard detected and points stored. Total images: {len(objpoints)}")
    cv2.imshow('Calibration', frame)
    key = cv2.waitKey(1)
    # Step 4: Break if more than 4 images have been stored
    if len(objpoints) > 4:
        print("Enough images captured for calibration.")
        break
    # Press ESC to exit early
    if key == 27:
        print("Calibration interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()

# Step 5: Calibrate the camera and save parameters
if len(objpoints) > 4:
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    # Save the parameters to an XML file
    filename = 'calibration.xml'
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("intrinsic", cameraMatrix)  # cameraMatrix
    fs.write("distortion", distCoeffs)   # distCoeffs
    fs.release()
    print(f"Calibration complete. Parameters saved to {filename}")
else:
    print("Not enough images were captured for calibration.")
