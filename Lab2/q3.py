import cv2
import numpy as np
import random


def check_label(label, eq):
    while eq[label] != label:
        label = eq[label]
    return label


def color(label_img):
    h, w = label_img.shape[:2]
    res = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate random colors for each unique label except 0 (background)
    unique_label = np.unique(label_img)
    to_color = {label: np.random.randint(0, 255, 3) for label in unique_label if label > 0}
    
    for i in range(h):
        for j in range(w):
            lab = label_img[i, j]
            if lab > 0:  # Only color the connected components
                res[i, j] = to_color[lab]
    
    return res


def cc(img):
    h, w = img.shape[:2]
    labeled = np.zeros_like(img, dtype=np.uint32)
    label = 1
    equiv = {}  # Dictionary to store equivalences
    
    # First pass: Label and record equivalences
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:  # Foreground pixel
                neighbors = []
                
                if i > 0 and labeled[i - 1, j] > 0:  # Top neighbor
                    neighbors.append(labeled[i - 1, j])
                if j > 0 and labeled[i, j - 1] > 0:  # Left neighbor
                    neighbors.append(labeled[i, j - 1])
                    
                if not neighbors:
                    labeled[i, j] = label
                    equiv[label] = label
                    label += 1
                elif len(neighbors) == 1 or equiv[neighbors[0]] == equiv[neighbors[1]]:
                    labeled[i, j] = equiv[neighbors[0]]
                else:
                    min_label = min(equiv[neighbors[0]], equiv[neighbors[1]])
                    max_label = max(equiv[neighbors[0]], equiv[neighbors[1]])
                    labeled[i, j] = min_label
                    for k in range(1, len(equiv) + 1):
                        if equiv[k] == max_label:
                            equiv[k] = min_label
                    
                    # for k in neighbors:
                    #     root = check_label(k, equiv)
                    #     equiv[root] = min_label
                        
    
    # # Second pass: Resolve equivalences
    # for l in equiv:
    #     equiv[l] = check_label(l, equiv)
    
    # Relabel the image based on resolved equivalences
    for i in range(h):
        for j in range(w):
            if labeled[i, j] > 0:
                labeled[i, j] = equiv[labeled[i, j]]
    
    return color(labeled)  # Return the colored labeled image


# Load the image in grayscale
image = cv2.imread("./result/q2.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(image, 117, 255, cv2.THRESH_BINARY)

# Apply connected component labeling and color the connected components
colored_res = cc(binary_image)

# Save and display the result
cv2.imwrite("./result/q3.jpg", colored_res)
cv2.imshow("Colored Connected Components", colored_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
