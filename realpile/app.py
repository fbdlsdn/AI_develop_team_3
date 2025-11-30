import streamlit as st
import cv2
import time
import threading

import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import keyboard  # q 입력으로 종료 확인

# -------------------------------
# 학습된 모델 로드
# -------------------------------
try:
    MODEL_FILE = "./DAiSEE/models/focus_model.pkl"
    if 'model' not in st.session_state:
        st.session_state['model'] = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"모델 파일이 로드되지 않았습니다. 경로를 확인해주세요: {MODEL_FILE}")
    st.session_state['model'] = None

# -------------------------------
# Mediapipe 초기화
# -------------------------------
mp_holistic = mp.solutions.holistic

# -------------------------------
# 민감도 기준 설정
# -------------------------------
CLOSED_THRESHOLD = 3         # 눈 완전히 감김 기준
HALF_CLOSED_THRESHOLD = 6    # 눈 반감김 기준
YAWN_THRESHOLD = 25          # 하품 기준
GAZE_THRESHOLD = 0.45        # 시선 이탈 기준
BLINK_MAX = 20               # 최대 깜빡임 수

# -------------------------------
# 집중도 계산 함수
# -------------------------------
def calculate_focus(yawn, blink, closed_time, half_closed_time, gaze_out_time,
                    w1=0.3, w2=0.2, w3=0.3, w4=0.2):
    """
    각 행동별 가중치를 곱해 패널티 계산 후 100에서 차감
    """
    penalty = 100 * (w1*yawn + w2*(blink/BLINK_MAX) + w3*(closed_time/10) + w4*(gaze_out_time/10))
    score = 100 - penalty
    return max(0, score)

# -------------------------------
# 집중도 저하 이유 분석
# -------------------------------
def analyze_focus_reason(blink, yawn, closed, half_closed, gaze):
    """
    평균값을 기반으로 가장 높은 패널티 항목을 원인으로 판단
    """
    penalties = {
        'blink': blink * 0.2,
        'yawn': yawn * 0.3,
        'closed': closed * 0.3,
        'half_closed': half_closed * 0.1,
        'gaze_out': gaze * 0.2
    }
    reason = max(penalties, key=penalties.get)
    return reason, penalties

# -------------------------------
# 웹캠 설정
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# 측정 변수 초기화
# -------------------------------
segment_start = time.time()
all_scores = []
all_states = []

blink_count = 0
yawn_count = 0
closed_seconds = 0
half_closed_seconds = 0
gaze_out_seconds = 0
eye_closed = False
yawn_state = False

# -------------------------------
# 평균 집중도 계산 및 낮은 원인 분석
# -------------------------------
if all_scores:
    avg_focus = round(sum(all_scores)/len(all_scores), 2)

    avg_blink = sum(s['blink_count'] for s in all_states)/len(all_states)
    avg_yawn = sum(s['yawn_count'] for s in all_states)/len(all_states)
    avg_closed = sum(s['closed_seconds'] for s in all_states)/len(all_states)
    avg_half_closed = sum(s['half_closed_seconds'] for s in all_states)/len(all_states)
    avg_gaze = sum(s['gaze_out_seconds'] for s in all_states)/len(all_states)

    reason, penalties = analyze_focus_reason(avg_blink, avg_yawn, avg_closed, avg_half_closed, avg_gaze)

    print(f"\n최종 평균 집중도: {avg_focus}")
    print(f"집중도가 낮은 주 원인: {reason}")
    print(f"상세 패널티: {penalties}")
else:
    print("집중도 데이터가 충분하지 않습니다.")

# ===============================
#  그래프 시각화 (matplotlib)
# ===============================
import matplotlib
matplotlib.use('TkAgg')     # OpenCV 충돌 방지
import matplotlib.pyplot as plt

if all_scores:
    # ---------- ① 시간 흐름에 따른 집중도 그래프 ----------
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(all_scores) + 1), all_scores, marker='o', linestyle='-', linewidth=2)
    plt.title("시간 흐름에 따른 집중도 변화")
    plt.xlabel("측정 구간 (10초 단위)")
    plt.ylabel("집중도 점수")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # ---------- ② 패널티 막대그래프 ----------
    labels = list(penalties.keys())
    values = list(penalties.values())
    main_cause = max(penalties, key=penalties.get)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    # 원인 막대 강조
    main_index = labels.index(main_cause)
    bars[main_index].set_edgecolor("red")
    bars[main_index].set_linewidth(3)

    # 막대 위 값 표시
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.title(f" 집중도 패널티 분석 (평균 집중도: {avg_focus})")
    plt.xlabel("행동 요소")
    plt.ylabel("패널티 크기")
    plt.ylim(0, max(values) + 0.2)
    plt.tight_layout()
    plt.show(block=True)

    print("그래프 출력 완료.")



def statistics_start(cap, stop_flag, model):
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True
    ) as holistic:
        try:
            while not stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break
                
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Streamlit Session State에서 변수 로드
                blink_count = st.session_state['blink_count']
                yawn_count = st.session_state['yawn_count']
                closed_seconds = st.session_state['closed_seconds']
                half_closed_seconds = st.session_state['half_closed_seconds']
                gaze_out_seconds = st.session_state['gaze_out_seconds']
                eye_closed = st.session_state['eye_closed']
                yawn_state = st.session_state['yawn_state']
                segment_start = st.session_state['segment_start']

                if results.face_landmarks:
                    landmarks = results.face_landmarks.landmark

                    # --- 눈 깜빡임 계산 ---
                    left_eye = abs(landmarks[145].y - landmarks[159].y) * h
                    right_eye = abs(landmarks[374].y - landmarks[386].y) * h
                    eye_avg = (left_eye + right_eye) / 2

                    if eye_avg < CLOSED_THRESHOLD:
                        closed_seconds += 1/30.0 # 프레임당 시간 누적
                        if not eye_closed:
                            eye_closed = True
                            blink_count += 1
                    else:
                        eye_closed = False

                    if CLOSED_THRESHOLD <= eye_avg < HALF_CLOSED_THRESHOLD:
                        half_closed_seconds += 1/30.0

                    # --- 하품 계산 ---
                    lip_dist = abs(landmarks[13].y - landmarks[14].y) * h
                    if lip_dist > YAWN_THRESHOLD:
                        if not yawn_state:
                            yawn_state = True
                            yawn_count += 1
                    else:
                        yawn_state = False

                    # --- 시선 이탈 계산 ---
                    left_center = (landmarks[33].x + landmarks[133].x)/2
                    right_center = (landmarks[362].x + landmarks[263].x)/2
                    eye_center_x = (left_center + right_center)/2
                    if not (0.5 - GAZE_THRESHOLD <= eye_center_x <= 0.5 + GAZE_THRESHOLD):
                        gaze_out_seconds += 1/30.0

                # -------------------------------
                # 10초 단위 집중도 예측 및 데이터 누적
                # -------------------------------
                if time.time() - segment_start >= 10:
                    
                    # 💡 모델이 없는 경우 대비
                    if model is not None:
                        features = np.array([[blink_count, yawn_count, closed_seconds, half_closed_seconds, gaze_out_seconds]])
                        score_pred = model.predict(features)[0]
                    else:
                        # 모델 없을 시 임시 점수 계산 함수 사용
                        score_pred = calculate_focus(yawn_count, blink_count, closed_seconds, half_closed_seconds, gaze_out_seconds)


                    st.session_state['all_scores'].append(score_pred)
                    st.session_state['all_states'].append({
                        'blink_count': blink_count,
                        'yawn_count': yawn_count,
                        'closed_seconds': closed_seconds,
                        'half_closed_seconds': half_closed_seconds,
                        'gaze_out_seconds': gaze_out_seconds
                    })

                    # 초기화
                    blink_count = yawn_count = closed_seconds = half_closed_seconds = gaze_out_seconds = 0
                    segment_start = time.time()
                
                # Streamlit Session State에 변수 저장
                st.session_state['blink_count'] = blink_count
                st.session_state['yawn_count'] = yawn_count
                st.session_state['closed_seconds'] = closed_seconds
                st.session_state['half_closed_seconds'] = half_closed_seconds
                st.session_state['gaze_out_seconds'] = gaze_out_seconds
                st.session_state['eye_closed'] = eye_closed
                st.session_state['yawn_state'] = yawn_state
                st.session_state['segment_start'] = segment_start
                
        finally:
            if cap is not None:
                cap.release()
            print("카메라 및 스레드 종료.")

# -------------------------------
# streamlit 상태 변수 초기화
# ------------------------------- 
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if 'recording_thread' not in st.session_state:
    st.session_state['recording_thread'] = None
if 'stop_flag' not in st.session_state:
    st.session_state['stop_flag'] = threading.Event()
if 'cap' not in st.session_state:
    st.session_state['cap'] = None

# -------------------------------
# 측정 변수 초기화
# ------------------------------- 
if 'all_scores' not in st.session_state:
    st.session_state['all_scores'] = []
if 'all_states' not in st.session_state:
    st.session_state['all_states'] = []

if 'segment_start' not in st.session_state:
    st.session_state['segment_start'] = time.time()
if 'blink_count' not in st.session_state:
    st.session_state['blink_count'] = 0
if 'yawn_count' not in st.session_state:
    st.session_state['yawn_count'] = 0
if 'closed_seconds' not in st.session_state:
    st.session_state['closed_seconds'] = 0
if 'half_closed_seconds' not in st.session_state:
    st.session_state['half_closed_seconds'] = 0
if 'gaze_out_seconds' not in st.session_state:
    st.session_state['gaze_out_seconds'] = 0
if 'eye_closed' not in st.session_state:
    st.session_state['eye_closed'] = False
if 'yawn_state' not in st.session_state:
    st.session_state['yawn_state'] = False
if 'last_analysis' not in st.session_state: # 최종 분석 결과를 저장할 공간
    st.session_state['last_analysis'] = None

st.title("공부 집중도 통계")
st.header("공부 집중도 통계")
st.caption("자신의 공부 시간을 기록하고 집중도 통계 산출하고 공부 습관을 점검해보세요!")

# 2. 버튼 클릭 시 호출될 함수 정의
def toggle_recording():
    # 현재 상태를 반전시킵니다. (False -> True 또는 True -> False)
    st.session_state['recording'] = not st.session_state['recording']

    if st.session_state['recording']:
        # --- 녹화 시작 로직 ---
        st.session_state['cap'] = cv2.VideoCapture(0)
        
        if not st.session_state['cap'].isOpened():
            st.error("카메라를 열 수 없습니다.")
            st.session_state['recording'] = False # 실패 시 상태 원복
            return
        
        # Stop 플래그 초기화
        st.session_state['stop_flag'].clear()

        # 별도의 스레드에서 run_camera 함수 실행
        st.session_state['recording_thread'] = threading.Thread(
            target=statistics_start, 
            args=(st.session_state['cap'], st.session_state['stop_flag'])
        )
        st.session_state['recording_thread'].start()
        
    else:
        # --- 녹화 중지 로직 ---
        if st.session_state['recording_thread'] is not None and st.session_state['recording_thread'].is_alive():
            # 스레드에 종료 신호 전달
            st.session_state['stop_flag'].set()
            # 스레드가 종료될 때까지 대기
            st.session_state['recording_thread'].join()
            st.session_state['recording_thread'] = None

# 3. 버튼 표시
# 버튼의 label을 현재 상태에 따라 변경합니다.
button_label = "통계 측정 중지" if st.session_state['recording'] else "통계 측정 시작"
st.button(button_label, on_click=toggle_recording)

# 4. 상태에 따른 텍스트 표시
if st.session_state['recording']:
    st.success("🔴 녹화중...") # st.success는 녹색 배경의 메시지를 표시합니다.
else:
    # 녹화 중이 아닐 때는 시작 안내 메시지를 표시할 수 있습니다.
    st.info("측정을 시작하려면 버튼을 눌러주세요.")