# Face Segmentation and Speaker Diarization Project

## 프로젝트 개요
이 프로젝트는 비디오에서 화자를 식별하고, 얼굴, 포즈, 손을 인식하여 비디오를 분석 및 시각화하는 프로젝트입니다. YOLO 모델을 사용한 사람 인식, Swin Transformer를 통한 이미지 처리, Mediapipe를 이용한 랜드마크 추출, Pyannote를 이용한 화자 다이어리제이션 등 다양한 기술이 사용되었습니다.

## 사용된 기술
- Python
- OpenCV
- Mediapipe
- YOLOv8
- Swin Transformer
- Pyannote.audio
- MoviePy
- Hugging Face Hub
- Concurrent Futures

## 프로젝트 구조
├── main.py
├── utils
│ ├── audio_processing.py
│ ├── huggingface_login.py
│ ├── logging_setup.py
│ ├── mediapipe_solutions.py
│ ├── pyannote_pipeline.py
│ ├── swin_model.py
│ ├── utils.py
│ ├── video_processing.py
│ └── yolo_model.py
└── README.md


## 설치 방법
1. Python 3.8 이상 버전을 설치합니다.
2. 가상 환경을 생성합니다.
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows

필수 패키지를 설치합니다.
pip install -r requirements.txt

실행 방법
Hugging Face 로그인 토큰을 준비합니다.
main.py 파일을 실행합니다.

python main.py

주요 모듈 및 함수 설명
main.py
프로젝트의 메인 스크립트로, 전체 파이프라인을 실행합니다.

utils/logging_setup.py
로깅 설정을 위한 모듈입니다.

setup_logger(): 로거를 설정하고 반환합니다.
utils/huggingface_login.py
Hugging Face에 로그인하기 위한 모듈입니다.

login_to_huggingface(token): Hugging Face에 로그인합니다.
utils/pyannote_pipeline.py
Pyannote Pipeline을 로드하고 설정합니다.

load_pyannote_pipeline(token): Pyannote Pipeline을 로드하고 설정합니다.
utils/yolo_model.py
YOLO 모델을 로드하고 설정합니다.

load_yolo_model(model_path): YOLO 모델을 로드합니다.
utils/swin_model.py
Swin Transformer 모델을 로드하고 설정합니다.

load_swin_model(): Swin Transformer 모델을 로드합니다.
utils/mediapipe_solutions.py
Mediapipe 솔루션을 로드합니다.

load_mediapipe_solutions(): Mediapipe 솔루션을 로드합니다.
utils/video_processing.py
비디오 프레임 처리 관련 함수가 포함되어 있습니다.

process_frame(frame, width, height): 프레임을 전처리합니다.
resize_frame(frame, width, height): 프레임의 크기를 조정합니다.
utils/audio_processing.py
오디오 처리 관련 함수가 포함되어 있습니다.

extract_audio(input_video_path, duration, output_audio_path): 비디오에서 오디오를 추출합니다.
merge_audio_with_video(video_path, audio_path, output_video_path): 비디오와 오디오를 병합합니다.
utils/utils.py
유틸리티 함수가 포함되어 있습니다.

get_mouth_aspect_ratio(landmarks): 입의 비율을 계산합니다.
get_movement_ratio(landmarks): 랜드마크의 움직임 비율을 계산합니다.
check_hand_over_mouth(face_landmarks, hand_landmarks): 손이 입 위에 있는지 확인합니다.
apply_highlight(frame, mask, color, alpha): 프레임에 하이라이트를 적용합니다.
