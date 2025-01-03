import cv2
import numpy as np
import matplotlib.pyplot as plt

#q1-a
def BGR_histogram(img):
    h, w = img.shape[:2]
    res = np.zeros((h, w, 3), dtype=np.uint8)
    
    B, G, R = cv2.split(img)
    
    histB, bins = np.histogram(B.flatten(), 256, [0, 256])
    histG, bins = np.histogram(G.flatten(), 256, [0, 256])
    histR, bins = np.histogram(R.flatten(), 256, [0, 256])
    
    cdfB = histB.cumsum()
    cdfG = histG.cumsum()
    cdfR = histR.cumsum()
    
    normalB = cdfB * 255 / cdfB.max()
    normalB = [round(normalB[i]) for i in range(len(normalB))]
    normalG = cdfG * 255 / cdfG.max()
    normalG = [round(normalG[i]) for i in range(len(normalG))]
    normalR = cdfR * 255 / cdfR.max()
    normalR = [round(normalR[i]) for i in range(len(normalR))]
    
    for i in range(h):
        for j in range(w):
            res[i][j][:] = (normalB[int(B[i][j])], normalG[int(G[i][j])], normalR[int(R[i][j])])
            
    cv2.imshow("result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            
    return res

    
    
    
image = cv2.imread("./histogram.jpg")
res = BGR_histogram(image)
cv2.imwrite("./result/q1a.jpg", res)

    