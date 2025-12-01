import streamlit as st
import cv2
import time
import threading

import cv2
import mediapipe as mp
import time
import numpy as np
import joblib

import matplotlib
matplotlib.use('TkAgg')     # OpenCV 충돌 방지
import matplotlib.pyplot as plt

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
if 'analysis' not in st.session_state: # 최종 분석 결과를 저장할 공간
    st.session_state['analysis'] = None

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
# 평균 집중도 계산 및 낮은 원인 분석
# -------------------------------
def analysis():
    all_scores = st.session_state['all_score']
    all_states = st.session_state['all_states']

    avg_focus = round(sum(all_scores)/len(all_scores), 2)

    avg_blink = sum(s['blink_count'] for s in all_states)/len(all_states)
    avg_yawn = sum(s['yawn_count'] for s in all_states)/len(all_states)
    avg_closed = sum(s['closed_seconds'] for s in all_states)/len(all_states)
    avg_half_closed = sum(s['half_closed_seconds'] for s in all_states)/len(all_states)
    avg_gaze = sum(s['gaze_out_seconds'] for s in all_states)/len(all_states)

    reason, penalties = analyze_focus_reason(avg_blink, avg_yawn, avg_closed, avg_half_closed, avg_gaze)

    st.session_state['analysis'] = {
        'avg_focus': avg_focus,
        'reason': reason,
        'penalties': penalties,
        'scores': all_scores
    }

    st.session_state['all_scores'] = []
    st.session_state['all_states'] = []
    st.session_state['segment_start'] = time.time()

# ===============================
#  그래프 시각화 (matplotlib)
# ===============================
def show_analysis(analysis_data):
    if not analysis_data:
        return 0
    
    st.markdown("---")
    st.header("📊 최종 집중도 분석 결과")
    
    avg_focus = analysis_data['avg_focus']
    reason = analysis_data['reason']
    penalties = analysis_data['penalties']
    all_scores = analysis_data['scores']

    st.metric(label="평균 집중도 점수", value=f"{avg_focus:.2f}점", delta_color="off")
    st.warning(f"주요 집중도 저하 원인: **{reason}**")

    ## 시간 흐름에 따른 집중도 그래프
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(1, len(all_scores) + 1), all_scores, marker='o', linestyle='-', linewidth=2, color='darkblue')
    ax1.set_title("시간 흐름에 따른 집중도 변화")
    ax1.set_xlabel("측정 구간 (10초 단위)")
    ax1.set_ylabel("집중도 점수")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig1)

    ## 패널티 막대그래프
    labels = list(penalties.keys())
    values = list(penalties.values())
    main_cause = max(penalties, key=penalties.get)
    main_index = labels.index(main_cause)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar(labels, values, color=['skyblue'] * len(labels))
    bars[main_index].set_color('red') # 주요 원인 강조

    # 막대 위 값 표시
    for i, v in enumerate(values):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)

    ax2.set_title("집중도 패널티 분석")
    ax2.set_xlabel("행동 요소")
    ax2.set_ylabel("패널티 크기")
    ax2.set_ylim(0, max(values) * 1.2)
    st.pyplot(fig2)



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


st.title("공부 집중도 통계")
st.header("공부 집중도 통계")
st.caption("자신의 공부 시간을 기록하고 집중도 통계 산출하고 공부 습관을 점검해보세요!")

# 2. 버튼 클릭 시 호출될 함수 정의
def toggle_recording():
    """녹화 시작/중지 로직과 분석 시작하는 함수"""
    
    # 중지 시, 통계 분석을 시작합니다.
    if st.session_state['recording']:
        # 현재 녹화 중 -> 중지 버튼 클릭
        if st.session_state['recording_thread'] is not None and st.session_state['recording_thread'].is_alive():
            st.session_state['stop_flag'].set()
            st.session_state['recording_thread'].join()
            st.session_state['recording_thread'] = None
        
        # 🟢 중요: 통계 분석 함수 호출 (녹화 종료 시점)
        analysis() 
        
    # 상태 반전
    st.session_state['recording'] = not st.session_state['recording']

    # 시작 시, 카메라 스레드를 시작합니다.
    if st.session_state['recording']:
        st.session_state['cap'] = cv2.VideoCapture(0)
        
        if not st.session_state['cap'].isOpened():
            st.error("카메라를 열 수 없습니다.")
            st.session_state['recording'] = False
            return
        
        st.session_state['stop_flag'].clear()
        st.session_state['recording_thread'] = threading.Thread(
            target=statistics_start, 
            args=(st.session_state['cap'], st.session_state['stop_flag'], st.session_state['model'])
        )
        st.session_state['recording_thread'].start()


# 3. 버튼 표시
button_label = "통계 측정 중지" if st.session_state['recording'] else "통계 측정 시작"
st.button(button_label, on_click=toggle_recording)

# 4. 상태에 따른 텍스트 표시
if st.session_state['recording']:
    st.success("🔴 녹화중... (백그라운드에서 집중도 측정 및 데이터 수집 중)") 
    
    # 실시간 측정 데이터 표시 (선택 사항)
    st.markdown("---")
    st.subheader("현재 측정 구간 (10초) 데이터")
    col1, col2, col3 = st.columns(3)
    col1.metric("깜빡임 수", f"{st.session_state['blink_count']:.0f}")
    col2.metric("하품 수", f"{st.session_state['yawn_count']:.0f}")
    col3.metric("눈 감음 시간", f"{st.session_state['closed_seconds']:.1f}초")
    st.info(f"누적 측정 구간: {len(st.session_state['all_scores'])}회")

else:
    st.info("측정을 시작하려면 버튼을 눌러주세요.")
    
    # 5. 최종 분석 결과 표시

    show_analysis(st.session_state['analysis'])
