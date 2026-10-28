# Author: Peiran (Cynthia) Jing | GitHub: https://github.com/pr000000
import sys, os, time      # system utilities: paths, timing
import cv2                # OpenCV: video capture and image processing
import torch              # PyTorch: model loading and inference
from pathlib import Path  # cross-platform path handling

# PyQt5 for the graphical interface
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Add the local YOLOv5 repo to the import path
YOLOV5_PATH = Path(__file__).resolve().parent / "yolov5"
sys.path.append(str(YOLOV5_PATH.parent))  # makes "import yolov5.xxx" resolvable

# YOLOv5 modules
from yolov5.models.common import DetectMultiBackend               # model loader
from yolov5.utils.general import non_max_suppression, scale_boxes  # NMS + box rescaling
from yolov5.utils.augmentations import letterbox                   # image preprocessing
from yolov5.utils.torch_utils import select_device                 # CPU/GPU selection


class GestureApp(QWidget):
    """Main window: live camera feed, detection results, confidence slider, screenshot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition System")
        self.resize(1100, 600)

        # Video display area
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        # Detection result panel
        self.result_label = QLabel("Results:")
        self.result_label.setFixedSize(300, 480)
        self.result_label.setStyleSheet("background-color: white; font-size: 20px; padding: 10px")

        # Slider for the confidence threshold (0-100 -> 0.00-1.00)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.slider_changed)

        self.slider_label = QLabel("Confidence: 0.20")  # shows the slider value

        # Screenshot button
        self.screenshot_button = QPushButton('📸 Screenshot')
        self.screenshot_button.clicked.connect(self.save_screenshot)

        # Control row (slider + screenshot button)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.screenshot_button)

        # Overall layout: video + results on top, controls below
        layout = QVBoxLayout()
        video_result_layout = QHBoxLayout()
        video_result_layout.addWidget(self.video_label)
        video_result_layout.addWidget(self.result_label)

        layout.addLayout(video_result_layout)
        layout.addLayout(control_layout)
        self.setLayout(layout)

        # Screenshot directory
        self.save_dir = Path("screenshots")
        self.save_dir.mkdir(exist_ok=True)

        # Initialise the YOLOv5 model
        self.device = select_device('')  # pick GPU if available, else CPU
        # Trained weights, resolved relative to this file (works on Windows/macOS/Linux)
        weights = Path(__file__).resolve().parent / "yolov5" / "runs" / "train" / "gesture_yolov5s_clean3" / "weights" / "best.pt"
        self.model = DetectMultiBackend(str(weights), device=self.device)
        self.model.eval()
        self.stride = self.model.stride
        self.imgsz = 640  # network input size
        self.class_names = ['Fist', 'OpenPalm', 'PeaceSign', 'ThumbsUp']
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (128, 0, 255)]  # one colour per class
        self.conf_thres = 0.5  # initial confidence threshold

        # Open the camera and refresh frames with a timer
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)  # call update_frame on every tick
        self.timer.start(30)  # every 30 ms (~33 FPS)

    def slider_changed(self, value):
        """Update the confidence threshold from the slider."""
        self.conf_thres = value / 100
        self.slider_label.setText(f"Confidence: {self.conf_thres:.2f}")

    def update_frame(self):
        """Grab one frame, run detection, draw boxes and update the UI."""
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame = frame.copy()  # keep the raw frame for screenshots

        # Preprocess: letterbox resize/pad, BGR -> RGB, HWC -> CHW, scale to [0, 1]
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).to(self.device).float() / 255.0
        img = img.unsqueeze(0)  # add batch dimension

        # Inference + non-maximum suppression
        with torch.no_grad():
            pred = self.model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=0.45)

        result_text = "Results:\n"
        for det in pred:
            if len(det):
                # Map boxes back to the original frame size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    cls = int(cls)
                    label = f"{self.class_names[cls]} {conf:.2f}"
                    color = self.colors[cls % len(self.colors)]
                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    result_text += label + "\n"

        # Update the result panel
        self.result_label.setText(result_text)

        # Convert the OpenCV frame into a Qt image and display it
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def save_screenshot(self):
        """Save the current raw frame as a JPEG."""
        save_path = self.save_dir / f'screenshot_{int(time.time())}.jpg'
        cv2.imwrite(str(save_path), self.frame)
        print(f"✅ Screenshot saved: {save_path}")

    def closeEvent(self, event):
        """Release the camera when the window closes."""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


# Launch the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GestureApp()
    win.show()
    sys.exit(app.exec_())
