import cv2
import numpy as np
import matplotlib.pyplot as plt

#q1-b
def HSV_histogram(img):
    h, w = img.shape[:2]
    res = np.zeros((h, w, 3), dtype=np.uint8)
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    
    histV, bins = np.histogram(V.flatten(), 256, [0, 256])
    cdfV = histV.cumsum()
    normalV = cdfV * 255 / cdfV.max()
    normalV = [round(normalV[i]) for i in range(len(normalV))]
    
    for i in range(h):
        for j in range(w):
            res[i][j][:] = (H[i][j], S[i][j], normalV[int(V[i][j])])
            
    result_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    
    cv2.imshow("HSV Histogram Equalized Image", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_bgr


image = cv2.imread("./histogram.jpg")
res = HSV_histogram(image)
cv2.imwrite("./result/q1b.jpg", res)
