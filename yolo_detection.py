# yolo_detection.py
from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # 원하는 YOLO 모델을 사용할 수 있습니다

    def detect_objects(self, image):
        results = self.model(image)
        return results

    def get_class_names(self, results):
        detections = []
        for result in results:
            # YOLO 결과 객체에서 클래스 이름 가져오기
            class_names = result.names if hasattr(result, 'names') else {}
            for box in result.boxes:
                class_id = int(box.cls.item())  # 클래스 ID를 정수로 변환
                class_name = class_names[class_id] if class_id in class_names else 'unknown'
                confidence = box.conf.item() if isinstance(box.conf, torch.Tensor) else box.conf
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()  # [x1, y1, x2, y2] 형태
                })
        return detections
