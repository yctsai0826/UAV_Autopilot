import cv2
import numpy as np

def blue_point(img):
    w = img.shape[0]
    h = img.shape[1]

    for i in range(w):
        for j in range(h):
            B, G, R = img[i][j]
            if(B > 100 and B * 0.6 > G and B * 0.6 > R):
                continue
            else:
                img[i][j][:] = (int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])) / 3
    
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def contrast_brightness(img):
    gray_img = np.array(img, dtype=np.int32)
    
    contrast = 100
    brightness = 40

    B, G, R = cv2.split(img)
    mask = (B + G) * 0.3 > R
    
    for i in range(3):
        gray_img[:, :, i][mask] = (gray_img[:, :, i][mask] - 127) * (contrast / 127 + 1) + 127 + brightness
        
    gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
    
    cv2.imshow("image", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gray_img


image = cv2.imread('./test.jpg')
res1 = blue_point(image)
cv2.imwrite('./result/1-1.jpg', res1)

image = cv2.imread('./test.jpg')
res1 = contrast_brightness(image)
cv2.imwrite('./result/1-2.jpg', res1)
