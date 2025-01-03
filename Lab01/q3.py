import cv2
import numpy as np

def edge(image):    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    ximg = cv2.filter2D(gray_img, -1, kernelx).astype(np.int32)
    yimg = cv2.filter2D(gray_img, -1, kernely).astype(np.int32)
    
    result = np.sqrt(ximg**2 + yimg**2)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


image = cv2.imread('./ive.jpg')
res2 = edge(image)
cv2.imwrite('./result/3.jpg', res2)
