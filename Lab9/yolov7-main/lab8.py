import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box

# 模型權重檔案路徑
WEIGHT = './runs/train/yolov7-lab08/weights/best.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加載模型
model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# 輸入影片檔案路徑
input_video_path = './lab08_test.mp4'
output_video_path = './output_video.mp4'

# 開啟影片檔案
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定影片寫出物件
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_orig = frame.copy()
    frame = letterbox(frame, (640, 640), stride=64, auto=True)[0]
    if device == "cuda":
        frame = transforms.ToTensor()(frame).to(device).half().unsqueeze(0)
    else:
        frame = transforms.ToTensor()(frame).to(device).float().unsqueeze(0)

    with torch.no_grad():
        output = model(frame)[0]
    output = non_max_suppression_kpt(output, 0.25, 0.65)[0]

    # 標註Label與Confidence
    output[:, :4] = scale_coords(frame.shape[2:], output[:, :4], frame_orig.shape).round()
    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, frame_orig, label=label, color=colors[int(cls)], line_thickness=1)

    # 寫出處理後的影格到輸出影片
    out.write(frame_orig)

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"處理完成，輸出影片儲存至: {output_video_path}")
