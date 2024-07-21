# RealSense Depth와 YOLO를 이용한 사물 인식 및 거리 측정

## 설명
이 Python 프로젝트는 RealSense Depth 카메라 기술과 PyTorch 및 Ultralytics가 지원하는 YOLO 객체 인식을 결합하여 실시간으로 사물을 인식하고 거리를 측정합니다. 감지된 객체 정보와 거리 데이터는 CSV 파일에 저장되어 추가 분석이 가능합니다.

## 사전 요구 사항
- Python 3.6 이상
- Intel RealSense SDK 2.0
- RealSense Depth 카메라
- PyTorch
- OpenCV
- NumPy
- Ultralytics

## 설치 방법
1. **저장소 클론**:

2. **가상 환경 생성**:
9   ```python3.10 -m venv realsense-env```


3. **가상 환경 활성화**:
    ```.\realsense-env\Scripts\activate```

4. **필요한 패키지 설치**:
    ```pip install -r requirements.txt```

5. **Intel RealSense SDK 설치**:

설치 가이드를 따라 Intel RealSense SDK를 시스템에 설치하세요.

## 사용법
다음 명령어를 실행하여 Python 스크립트를 실행하세요:
    ```python main.py```
