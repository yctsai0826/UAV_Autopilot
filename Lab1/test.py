import cv2
import numpy as np
import math

img = cv2.imread('test.jpg')

height, width = img.shape[:2]

result_img=np.zeros((height,width,3),np.uint8)


for i in range(height):
    for j in range(width):
        if img[i][j][0] > 100 and img[i][j][0]*0.6 > img[i][j][1] and img[i][j][0]*0.6 > img[i][j][2]:
            result_img[i][j] = img[i][j]
        else:
            print(img[i][j][0],img[i][j][1],img[i][j][2])
            print(int(((img[i][j][0])+(img[i][j][1])+(img[i][j][2]))/3))
            result_img[i][j][:] = int((int(img[i][j][0])+int(img[i][j][1])+int(img[i][j][2]))/3)

cv2.imshow('1-1 result',result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('1_1_result.jpg',result_img)