import cv2
import numpy as np

# Open webcam feed
cap = cv2.VideoCapture(1)

# Get a frame to obtain the source image dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    exit()
h_s, w_s, _ = frame.shape

# Define the source points (corners of the webcam image)
cap_corner = np.float32([
    [0, 0],
    [w_s - 1, 0],
    [w_s - 1, h_s - 1],
    [0, h_s - 1]
])

# Create destination image (e.g., a blank image)
destination_image = cv2.imread('screen.jpg')
h_d, w_d = destination_image.shape[:2]

# Define the destination points (corners of the quadrilateral)
img_corner = np.float32([[413, 865], [1634, 217], [1645, 1253], [335, 1409]])

# Compute the homography matrix
H = cv2.getPerspectiveTransform(cap_corner, img_corner)
H_inv = np.linalg.inv(H)

# Create a mask for the destination quadrilateral
mask = np.zeros((h_d, w_d), dtype=np.uint8)
cv2.fillConvexPoly(mask, img_corner.astype(np.int32), 1)

# Indices of points inside the quadrilateral
ys, xs = np.where(mask == 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for x_d, y_d in zip(xs, ys):

        dst_pt = np.array([x_d, y_d, 1])

        src_pt = H_inv @ dst_pt
        src_pt = src_pt / src_pt[2]
        x_s, y_s = src_pt[0], src_pt[1]
        
        if 0 <= x_s < w_s - 1 and 0 <= y_s < h_s - 1:
            # Bilinear interpolation
            x0 = int(np.floor(x_s))
            x1 = x0 + 1
            y0 = int(np.floor(y_s))
            y1 = y0 + 1
            dx = x_s - x0
            dy = y_s - y0
            
            # Get pixel values
            I00 = frame[y0, x0].astype(np.float32)
            I01 = frame[y0, x1].astype(np.float32)
            I10 = frame[y1, x0].astype(np.float32)
            I11 = frame[y1, x1].astype(np.float32)
            
            # Bilinear Interpolate
            Ixy = (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I01 \
                + (1 - dx) * dy * I10 + dx * dy * I11
                
            destination_image[y_d, x_d] = Ixy.astype(np.uint8)

    cv2.imshow('Warped Image', destination_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
