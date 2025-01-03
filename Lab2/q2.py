import cv2
import numpy as np

#q2
def otsu_threshold(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    total_pixels = img.shape[0] * img.shape[1]

    current_max = 0
    threshold = 0
    sum_all_pixels = np.sum([i * hist[i] for i in range(256)])
    weight_bg = 0
    sum_bg = 0

    for t in range(256):
        weight_bg += hist[t]    # cdf
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break
        
        sum_bg += t * hist[t]
        
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all_pixels - sum_bg) / weight_fg
        
        between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = t

    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Otsu Thresholded Image", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return threshold, binary_img


image = cv2.imread('./input.jpg', cv2.IMREAD_GRAYSCALE)
threshold, binary_image = otsu_threshold(image)
print(f"Threshold: {threshold}")
cv2.imwrite("./result/q2.jpg", binary_image)
