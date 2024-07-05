import cv2
import torch
import os
import concurrent.futures
from collections import deque

from utils.logging_setup import setup_logger
from utils.huggingface_login import login_to_huggingface
from utils.pyannote_pipeline import load_pyannote_pipeline
from utils.yolo_model import load_yolo_model
from utils.swin_model import load_swin_model
from utils.mediapipe_solutions import load_mediapipe_solutions
from utils.video_processing import process_frame, resize_frame
from utils.audio_processing import extract_audio, merge_audio_with_video
from utils.utils import get_mouth_aspect_ratio, get_movement_ratio, check_hand_over_mouth, apply_highlight

logger = setup_logger()

# Hugging Face 로그인
huggingface_token = "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"
login_to_huggingface(huggingface_token)

# 모델 및 솔루션 로드
pipeline = load_pyannote_pipeline(huggingface_token)
yolo_model = load_yolo_model('C:/Users/Cho-User/Desktop/HJ/Project/face_segmentation/yolov8x-seg.pt')
swin_model = load_swin_model()
face_mesh, pose, hands = load_mediapipe_solutions()

# 비디오 파일 경로
input_video_path = 'C:/Users/Cho-User/Desktop/HJ/Project/face_segmentation/t1/비둘기랑  대화하는 임다혜 ㅋㅋㅋㅋ 신병.mp4'
output_video_path = 'C:/Users/Cho-User/Desktop/HJ/Project/face_segmentation/t1/output.mp4'

# 비디오 읽기
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

# 해상도 축소
scale = 0.5
width = int(width * scale)
height = int(height * scale)

# VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 처음 10초까지만 사용
max_duration = min(duration, 10)
max_frame_number = int(max_duration * fps)

# 음성 파일 추출 및 화자 다이어리제이션
audio_path = extract_audio(input_video_path, max_duration, "temp_audio.wav")
diarization = pipeline(audio_path)
logger.info("Performed speaker diarization")

# 화자 다이어리제이션 결과를 미리 저장
speaker_intervals = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]

def get_speakers(frame_time, speaker_intervals):
    speakers = set()
    for start, end, speaker in speaker_intervals:
        if start <= frame_time <= end:
            speakers.add(speaker)
    return speakers

MOUTH_WEIGHT = 0.5
FACE_WEIGHT = 0.3
BODY_WEIGHT = 0.2

MOUTH_AR_THRESH = 0.1  # Threshold for mouth aspect ratio to detect speaking
MOVEMENT_THRESH = 0.3  # Threshold for movement ratio to detect speaking
frame_number = 0

# 화자별로 추적하기 위한 데이터 구조
active_speakers = {}

# Track previous frames to handle occlusions and other challenges
previous_frames = deque(maxlen=5)

# 최대 인원수 추적을 위한 변수
max_person_count = 0

# ThreadPoolExecutor 생성
executor = concurrent.futures.ThreadPoolExecutor()

while cap.isOpened() and frame_number < max_frame_number:
    ret, frame = cap.read()
    if not ret:
        break

    # 해상도 축소
    frame = resize_frame(frame, width, height)

    try:
        # Swin Transformer 입력 준비
        swin_input = cv2.resize(frame, (224, 224))
        swin_input = torch.from_numpy(swin_input).permute(2, 0, 1).unsqueeze(0).float().to(device)  # N, C, H, W

        # Mixed Precision 적용
        with torch.cuda.amp.autocast():
            swin_output = swin_model(swin_input)

        # 메모리 정리
        del swin_input
        torch.cuda.empty_cache()

        frame_time = frame_number / fps
        current_speakers = get_speakers(frame_time, speaker_intervals)

        results = None  # 초기화
        face_results = None
        pose_results = None
        hand_results = None
        processed_frame = frame  # 초기화

        if current_speakers:
            yolo_future = executor.submit(yolo_model.track, source=frame, tracker='C:/Users/Cho-User/Desktop/HJ/Project/face_segmentation/t1/bytetrack.yaml')
            results = yolo_future.result()

            # YOLO의 인식 결과로 최대 인원수 갱신
            person_count = sum(1 for result in results if result.boxes is not None)
            max_person_count = max(max_person_count, person_count)

            # Mediapipe 처리는 YOLO가 사람을 인식한 경우에만 수행
            if any(result.boxes is not None for result in results):
                process_future = executor.submit(process_frame, frame, width, height)
                face_results, pose_results, hand_results, processed_frame = process_future.result()

            best_speaker_score = 0
            best_speaker_box = None
            best_speaker_mask = None
            person_found = False

            for speaker in current_speakers:
                if speaker not in active_speakers:
                    active_speakers[speaker] = {'face_landmarks': [], 'pose_landmarks': [], 'hand_landmarks': []}

                if face_results and face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        face_landmarks = [(landmark.x * width, landmark.y * height) for landmark in face_landmarks.landmark]
                        active_speakers[speaker]['face_landmarks'].append(face_landmarks)

                if pose_results and pose_results.pose_landmarks:
                    pose_landmarks = [(landmark.x * width, landmark.y * height) for landmark in pose_results.pose_landmarks.landmark]
                    active_speakers[speaker]['pose_landmarks'].append(pose_landmarks)

                if hand_results and hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        hand_landmarks = [(landmark.x * width, landmark.y * height) for landmark in hand_landmarks.landmark]
                        active_speakers[speaker]['hand_landmarks'].append(hand_landmarks)

                # 마지막 프레임의 랜드마크를 사용하여 움직임 계산
                face_landmarks = active_speakers[speaker]['face_landmarks'][-1] if active_speakers[speaker]['face_landmarks'] else []
                pose_landmarks = active_speakers[speaker]['pose_landmarks'][-1] if active_speakers[speaker]['pose_landmarks'] else []
                hand_landmarks = active_speakers[speaker]['hand_landmarks'][-1] if active_speakers[speaker]['hand_landmarks'] else []

                mouth_movement_ratio = get_mouth_aspect_ratio(face_landmarks) if face_landmarks else 0
                face_movement_ratio = get_movement_ratio(face_landmarks)
                body_movement_ratio = get_movement_ratio(pose_landmarks)

                hand_over_mouth = check_hand_over_mouth(face_landmarks, hand_landmarks)

                # 어텐션 메커니즘을 통한 종합 움직임 점수 계산
                total_movement_score = (
                    MOUTH_WEIGHT * mouth_movement_ratio +
                    FACE_WEIGHT * face_movement_ratio +
                    BODY_WEIGHT * body_movement_ratio
                )

                # 입의 움직임이 없으면 화자로 감지하지 않음
                if (mouth_movement_ratio > MOUTH_AR_THRESH or hand_over_mouth):
                    for result in results:
                        if result.boxes is not None and result.masks is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            scores = result.boxes.conf.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()
                            masks = result.masks.data.cpu().numpy()

                            for box, score, cls, mask in zip(boxes, scores, classes, masks):
                                if int(cls) == 0:  # 'person' class id
                                    x1, y1, x2, y2 = box
                                    mask_resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

                                    if total_movement_score > MOVEMENT_THRESH and (not face_landmarks or (x1 <= face_landmarks[0][0] <= x2 and y1 <= face_landmarks[0][1] <= y2)):
                                        if total_movement_score > best_speaker_score:
                                            best_speaker_score = total_movement_score
                                            best_speaker_box = (x1, y1, x2, y2)
                                            best_speaker_mask = mask_resized
                                        person_found = True

            if person_found and best_speaker_box and best_speaker_mask is not None:
                x1, y1, x2, y2 = best_speaker_box
                processed_frame = apply_highlight(processed_frame, best_speaker_mask, [0, 255, 0], alpha=0.5)  # 반투명 하이라이트 적용
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Speaker Score: {best_speaker_score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(processed_frame)
        else: 
            out.write(frame)

        frame_number += 1

        if frame_number % 100 == 0:
            logger.info(f"Processed frame {frame_number}")

        # 메모리 정리
        del swin_output
        if face_results:
            del face_results
        if pose_results:
            del pose_results
        if hand_results:
            del hand_results
        if processed_frame is not frame:
            del processed_frame
        if results:
            del results
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error processing frame {frame_number}: {e}")
        continue

cap.release()
out.release()
executor.shutdown()

# 임시 오디오 파일을 비디오에 병합
merge_audio_with_video(output_video_path, "temp_audio.wav", "final_output.mp4")

# 임시 오디오 파일 삭제
os.remove("temp_audio.wav")
logger.info("Deleted temporary audio file")
