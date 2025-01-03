import cv2
import numpy as np
import math


def bilinear_interpolation(img):
    height, width = img.shape[:2]
    result_img = np.zeros((height*3, width*3, 3), np.uint8)

    for i in range(height*3):
        for j in range(width*3):#
            x = j/3
            y = i/3

            x1 = math.floor(x)
            x2 = min(x1+1, width-1)
            y1 = math.floor(y)
            y2 = min(y1+1, height-1)

            f_x_y1 = (x2-x)*img[y1][x1] + (x-x1)*img[y1][x2]
            f_x_y2 = (x2-x)*img[y2][x1] + (x-x1)*img[y2][x2]
            result_img[i][j] = (y2-y)*f_x_y1 + (y-y1)*f_x_y2
    
    cv2.imshow('image', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result_img


image = cv2.imread('ive.jpg')
res2 = bilinear_interpolation(image)
cv2.imwrite('result/2.jpg', res2)
