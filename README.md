프로젝트 개요
이 프로젝트는 영상에서 얼굴과 신체 포즈를 분석하여 화자를 식별하고, 해당 화자에게 하이라이트를 적용하는 시스템입니다. YOLO, Swin Transformer, Mediapipe, Pyannote 등의 다양한 최신 기술을 통합하여 구현되었습니다.

설치 및 실행 방법
1. 필수 라이브러리 설치
2. 다음 명령어를 사용하여 필요한 라이브러리를 설치합니다:

![image](https://github.com/zodia8393/Speaker_Highlignt/assets/83521479/fe753529-40dd-47eb-9b04-5eea09c0ae92)

2. 프로젝트 구조
프로젝트 디렉토리는 다음과 같은 구조를 가집니다:
![image](https://github.com/zodia8393/Speaker_Highlignt/assets/83521479/bffbffb1-3d6d-4feb-8b3d-e9a70717161d)

3. 환경 변수 설정
Hugging Face 인증 토큰을 utils/huggingface_login.py 파일에 설정합니다:
![image](https://github.com/zodia8393/Speaker_Highlignt/assets/83521479/ecd4cc42-025b-4b83-9132-b82bc71d33e8)

4. YOLO 모델 경로 설정
YOLO 모델 경로를 utils/yolo_model.py 파일에 설정합니다:
![image](https://github.com/zodia8393/Speaker_Highlignt/assets/83521479/4f578491-d4e5-40e7-8a87-07fa00f0f438)

5. 비디오 파일 경로 설정
분석할 비디오 파일 경로를 main.py 파일에 설정합니다:
![image](https://github.com/zodia8393/Speaker_Highlignt/assets/83521479/da756ae8-dbe1-4036-9ead-e60791d9bcdb)

6. 실행
다음 명령어를 사용하여 메인 스크립트를 실행합니다:
![image](https://github.com/zodia8393/Speaker_Highlignt/assets/83521479/3678a781-1ab7-4150-8a35-06649b16d067)

기능 설명
화자 식별: Pyannote를 사용하여 비디오에서 화자를 식별합니다.
얼굴 및 포즈 추출: Mediapipe를 사용하여 얼굴 랜드마크와 신체 포즈를 추출합니다.
하이라이트 적용: 식별된 화자에게 YOLO를 통해 하이라이트를 적용합니다.
오디오 처리: MoviePy를 사용하여 비디오에서 오디오를 추출하고, 화자 정보와 함께 최종 비디오를 생성합니다.

문의
프로젝트와 관련된 문의는 이메일 (chohj_1019@naver.com)로 연락 주시기 바랍니다.
