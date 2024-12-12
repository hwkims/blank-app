import streamlit as st
import cv2
import pygame
import threading
import time
from ultralytics import YOLO
import numpy as np

# YOLOv8 모델 로드 (학습한 모델 파일 경로)
model = YOLO('trained_model.pt')

# 알림 소리 파일 경로 (소리 파일을 프로젝트 폴더에 넣고 사용)
alert_sound = 'alert_sound.mp3'

# pygame 초기화
pygame.mixer.init()

# 소리 재생을 위한 비동기 함수
def play_alert_sound():
    while True:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
        print("소리 재생됨!")  # 소리 재생 시 출력
        time.sleep(1)  # 너무 자주 반복되지 않도록 잠시 대기

# 소리 재생을 위한 스레드 시작
alert_thread = threading.Thread(target=play_alert_sound, daemon=True)
alert_thread.start()

# 웹캠 열기 (기본 카메라 0번 사용)
cap = cv2.VideoCapture(0)

# 현재 소리가 재생 중인지 확인
is_playing = False

# Streamlit 웹 페이지 설정
st.title("Cigarette Detection System")
st.subheader("웹캠을 통해 담배를 탐지하고 알림 소리를 재생합니다.")
frame_placeholder = st.empty()  # 프레임을 표시할 공간 예약

# 실시간으로 웹캠 영상을 스트리밍하는 함수
def process_video():
    global is_playing
    
    while True:
        ret, frame = cap.read()  # 웹캠으로부터 프레임 읽기

        if not ret:
            st.error("웹캠을 열 수 없습니다.")
            break

        # YOLOv8 모델로 객체 탐지
        results = model(frame)

        # 결과에서 탐지된 객체 확인
        cigarette_detected = False

        # results.pred[0]는 YOLO 모델이 반환하는 첫 번째 이미지에 대한 탐지 결과입니다
        for result in results[0].boxes:
            # result.xyxyn[5]는 탐지된 클래스 ID입니다.
            class_id = int(result.cls.item())  # 결과 클래스 ID 추출
            if model.names[class_id] == 'cigarette':  # 'cigarette' 클래스가 탐지되었는지 확인
                cigarette_detected = True
                break

        # 'cigarette'가 탐지되었을 때
        if cigarette_detected:
            print("Cigarette detected!")  # 탐지 시 출력
            if not is_playing:  # 소리가 재생 중이 아니면
                pygame.mixer.music.set_volume(1.0)  # 볼륨을 최대값으로 설정
                is_playing = True
        else:
            # 'cigarette'가 탐지되지 않았을 때
            print("No cigarette detected!")  # 탐지되지 않으면 출력
            pygame.mixer.music.set_volume(0.0)  # 볼륨을 0으로 설정
            is_playing = False

        # 탐지된 결과를 영상에 표시
        annotated_frame = results[0].plot()  # 탐지된 객체 표시

        # OpenCV에서 읽은 이미지를 Streamlit에서 표시할 수 있도록 변환
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # 이미지 표시
        frame_placeholder.image(frame_rgb)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Streamlit UI에서 영상 스트리밍을 시작하는 버튼
if st.button("Start Video"):
    process_video()

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
