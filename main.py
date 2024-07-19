import cv2
import time
import os
import csv
from realsense_depth import RealSenseCamera
from yolo_detection import YOLODetector

def get_next_filename(base_name, ext):
    """기존 파일 이름과 중복되지 않도록 새로운 파일 이름을 생성합니다."""
    counter = 1
    filename = f"{base_name}{counter}.{ext}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}{counter}.{ext}"
    return filename

def main():
    # RealSense 카메라 및 YOLO 모델 초기화
    camera = RealSenseCamera()
    detector = YOLODetector()

    # 데이터 저장을 위한 파일 이름 설정
    base_filename = "data"
    filename = get_next_filename(base_filename, 'csv')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # CSV 파일 헤더 작성
        writer.writerow([
            'Timestamp', 'Class Name', 'Confidence', 
            'Bounding Box (x1, y1, x2, y2)', 'Center (x, y)', 'Depth (mm)'
        ])

        try:
            while True:
                # 프레임 캡처 및 객체 탐지
                color_image, depth_image = camera.get_frames()
                results = detector.detect_objects(color_image)
                detections = detector.get_class_names(results)

                # 현재 시간 기록
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                # 데이터 기록
                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection['box'])
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # 중심 좌표 계산
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 중심 좌표의 깊이 값 추출
                    depth = depth_image[cy, cx] if (0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]) else 0
                    
                    # CSV 파일에 기록할 데이터
                    writer.writerow([
                        current_time, class_name, f"{confidence:.2f}", 
                        f"({x1}, {y1}), ({x2}, {y2})", f"({cx}, {cy})", depth
                    ])

                    # 물체 바운딩 박스와 정보를 화면에 표시
                    color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({confidence:.2f}) Depth: {depth}mm"
                    color_image = cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 결과 이미지 표시
                cv2.imshow('RealSense', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # CSV 파일이 일정 크기를 초과하면 새로운 파일 생성
                file.flush()  # 버퍼링된 데이터를 파일에 기록
                if os.path.getsize(filename) > 1024 * 1024:  # 예를 들어 1MB 초과 시
                    file.close()
                    filename = get_next_filename(base_filename, 'csv')
                    file = open(filename, 'w', newline='')
                    writer = csv.writer(file)
                    writer.writerow([
                        'Timestamp', 'Class Name', 'Confidence', 
                        'Bounding Box (x1, y1, x2, y2)', 'Center (x, y)', 'Depth (mm)'
                    ])

        finally:
            camera.release()
            cv2.destroyAllWindows()
            file.close()

if __name__ == "__main__":
    main()
