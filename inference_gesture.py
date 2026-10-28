# Author: Peiran (Cynthia) Jing | GitHub: https://github.com/pr000000
import os
import sys
import torch
import cv2
from pathlib import Path

# Project root = folder containing this script; YOLOv5 is cloned into ./yolov5
ROOT = Path(__file__).resolve().parent
YOLOV5_PATH = ROOT / "yolov5"
sys.path.append(str(YOLOV5_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# ====== Configuration ======
weights_path = YOLOV5_PATH / "runs" / "train" / "gesture_yolov5s_clean3" / "weights" / "best.pt"
imgsz = 640
conf_thres = 60  # initial value in percent (0-100)
iou_thres = 45   # IoU threshold for NMS, in percent
device = select_device('')
class_names = ['Fist', 'OpenPalm', 'PeaceSign', 'ThumbsUp']
colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (128, 0, 255)]

save_dir = ROOT / 'screenshots'
save_dir.mkdir(parents=True, exist_ok=True)

# ====== Model initialisation ======
model = DetectMultiBackend(str(weights_path), device=device)
stride = model.stride
model.eval()

print(f"✅ Model loaded on {device}. Use ↑/↓ to adjust the threshold, current: {conf_thres / 100:.2f}")

# ====== Window and trackbar ======
WINDOW = "Gesture Recognition"
cv2.namedWindow(WINDOW)
cv2.createTrackbar("Confidence", WINDOW, conf_thres, 100, lambda x: None)

# ====== Open the camera ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open the camera")
    sys.exit(1)

print("✅ Recognition started. Press 'q' to quit, 's' to take a screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read the live trackbar value (percent -> fraction)
    conf_thres = cv2.getTrackbarPos("Confidence", WINDOW)
    conf_thres_float = conf_thres / 100

    # ==== Preprocessing: letterbox, BGR->RGB, HWC->CHW, scale to [0, 1] ====
    img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img.copy()).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # ==== Inference + NMS ====
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_thres_float, iou_thres=iou_thres / 100)

    # ==== Draw results ====
    for det in pred:
        if len(det):
            # Rescale boxes from network input size to the original frame
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                cls = int(cls)
                label = f"{class_names[cls]} {conf:.2f}"
                color = colors[cls % len(colors)]
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame and handle keyboard input
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = save_dir / f'screenshot_{cv2.getTickCount()}.jpg'
        cv2.imwrite(str(filename), frame)
        print(f"🖼️ Screenshot saved: {filename}")
    elif key == 82:  # ↑ key: raise threshold
        conf_thres = min(100, conf_thres + 5)
        cv2.setTrackbarPos("Confidence", WINDOW, conf_thres)
    elif key == 84:  # ↓ key: lower threshold
        conf_thres = max(0, conf_thres - 5)
        cv2.setTrackbarPos("Confidence", WINDOW, conf_thres)

cap.release()
cv2.destroyAllWindows()
