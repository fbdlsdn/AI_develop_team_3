import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# -------------------------------
# 설정 및 Mediapipe 초기화
# -------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
FACE_MESH_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

# -------------------------------
# 민감도 기준 설정
# -------------------------------
CLOSED_THRESHOLD = 3      
HALF_CLOSED_THRESHOLD = 6   
YAWN_THRESHOLD = 25       
GAZE_THRESHOLD = 0.45     
BLINK_MAX = 20            

# -------------------------------
# 집중 단계 정의 및 계산 함수 (유지)
# -------------------------------
def get_focus_stage(score):
    if score >= 75:
        return 3, "최상 (Excellent)"
    elif score >= 50:
        return 2, "양호 (Good)"
    elif score >= 25:
        return 1, "보통 (Normal)"
    else:
        return 0, "집중 저하 (Low)"

def calculate_focus_score(yawn, blink, closed_time, half_closed_time, gaze_out_time,
                          w1=0.3, w2=0.2, w3=0.3, w4=0.2):
    blink_normalized = min(blink / BLINK_MAX, 1.0) 
    closed_normalized = min(closed_time / 10.0, 1.0) 
    gaze_out_normalized = min(gaze_out_time / 10.0, 1.0) 
    
    penalty_ratio = (w1 * yawn) + (w2 * blink_normalized) + \
                    (w3 * closed_normalized) + (w4 * gaze_out_normalized)
    score = 100 - (100 * penalty_ratio)
    return max(0, score)

def analyze_focus_reason(blink, yawn, closed, half_closed, gaze):
    blink_normalized = min(blink / BLINK_MAX, 1.0)
    closed_normalized = min(closed / 10.0, 1.0)
    half_closed_normalized = min(half_closed / 10.0, 1.0)
    gaze_out_normalized = min(gaze / 10.0, 1.0)
    
    penalties = {
        '하품 (Yawn)': yawn * 0.3, 
        '눈 완전히 감음 (Closed)': closed_normalized * 0.3, 
        '시선 이탈 (Gaze-out)': gaze_out_normalized * 0.2, 
        '깜빡임 (Blink)': blink_normalized * 0.2, 
        '눈 반쯤 감음 (Half-Closed)': half_closed_normalized * 0.1, 
    }
    reason = max(penalties, key=penalties.get)
    total_penalty = sum(penalties.values())
    
    if total_penalty > 0:
        penalties_ratio = {k: (v / total_penalty) * 100 for k, v in penalties.items()}
    else:
        penalties_ratio = {k: 0 for k in penalties.keys()}
        
    return reason, penalties_ratio

# -------------------------------
# 결과 표시 함수 (재사용을 위해 분리)
# -------------------------------
def display_results(scores, states, total_time_segments, is_final=False):
    """집중도 결과 (메트릭, 그래프, 원인 분석)을 표시하는 함수"""
    
    if not scores:
        return

    # 1. 최종 메트릭 계산
    total_segments = len(scores)
    avg_focus_score = round(sum(scores)/total_segments, 2)
    stage_level, stage_desc = get_focus_stage(avg_focus_score)
    total_time = f"{total_time_segments * 10} 초"

    metric_title = "✨ 최종 분석 결과" if is_final else "🔍 저장된 기록 분석"
    
    st.subheader(metric_title)
    
    st_metrics = st.columns(3)
    st_metrics[0].metric("평균 집중도", f"{avg_focus_score} 점")
    st_metrics[1].metric("최종 집중 단계", f"단계 {stage_level}", stage_desc)
    st_metrics[2].metric("총 측정 시간", total_time)

    # 2. 집중도 변화 그래프
    st.header("📊 시간 흐름에 따른 집중도 변화")
    df_scores = pd.DataFrame({
        '측정 구간': range(1, total_segments + 1),
        '집중도 점수': scores
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_scores['측정 구간'], df_scores['집중도 점수'], marker='o', linestyle='-', linewidth=2, color='skyblue')
    ax.set_title("10초 단위 집중도 변화 (수식 기반)")
    ax.set_xlabel("측정 구간 (10초)")
    ax.set_ylabel("집중도 점수 (0-100)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    st.pyplot(fig)
         
    st.markdown("---")
    
    # 3. 패널티 분석 그래프
    st.header("📉 집중도 저하 원인 분석")
    
    avg_blink = sum(s['blink_count'] for s in states) / total_segments
    avg_yawn = sum(s['yawn_count'] for s in states) / total_segments
    avg_closed = sum(s['closed_seconds'] for s in states) / total_segments
    avg_half_closed = sum(s['half_closed_seconds'] for s in states) / total_segments
    avg_gaze = sum(s['gaze_out_seconds'] for s in states) / total_segments

    reason, penalties_ratio = analyze_focus_reason(avg_blink, avg_yawn, avg_closed, avg_half_closed, avg_gaze)
    
    st.subheader(f"⚡️ 주요 집중 저하 원인: **{reason}**")
    
    labels = list(penalties_ratio.keys())
    values = list(penalties_ratio.values())
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar(labels, values, color=['lightcoral' if l == reason else 'lightblue' for l in labels])
    
    ax2.set_title("집중도 패널티 기여도 (%)")
    ax2.set_xlabel("행동 요소")
    ax2.set_ylabel("패널티 기여 비율 (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y')
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom')

    st.pyplot(fig2)


# -------------------------------
# Streamlit 메인 함수
# -------------------------------
def main():
    st.title("🧠 실시간 집중도 측정 애플리케이션 (화면 비노출 모드)")
    
    st.info("💡 **웹캠은 백그라운드에서 실시간 측정을 위해 활성화됩니다.** 카메라 영상은 사용자에게 표시되지 않고, 10초마다 집중도 분석 결과만 업데이트됩니다.")

    # -------------------------------
    # 상태 초기화 및 History 추가
    # -------------------------------
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'all_scores' not in st.session_state:
        st.session_state.all_scores = []
    if 'all_states' not in st.session_state:
        st.session_state.all_states = []
    if 'recording_start_time' not in st.session_state:
        st.session_state.recording_start_time = 0
    if 'current_avg_score' not in st.session_state:
        st.session_state.current_avg_score = 0
    if 'current_stage_desc' not in st.session_state:
        st.session_state.current_stage_desc = "측정 전"
    if 'current_stage_level' not in st.session_state:
        st.session_state.current_stage_level = 0
    if 'current_total_time' not in st.session_state:
        st.session_state.current_total_time = "00:00:00"
    if 'selected_history_index' not in st.session_state:
        st.session_state.selected_history_index = None # 선택된 기록 인덱스 (None: 현재 세션)
    if 'history' not in st.session_state:
        st.session_state.history = [] # 완료된 녹화 기록 저장

    # -------------------------------
    # 사이드바 (기록 목록 표시)
    # -------------------------------
    with st.sidebar:
        st.header("📚 측정 기록")
        
        # '현재 기록 보기' 버튼 (녹화 중이 아니거나 기록이 있을 때)
        if st.session_state.is_running or st.session_state.all_scores or st.session_state.history:
            is_current_active = st.session_state.selected_history_index is None
            if st.button("▶️ 현재 세션 기록 보기", disabled=is_current_active):
                st.session_state.selected_history_index = None
                st.rerun()

        st.markdown("---")
        
        if st.session_state.history:
            st.subheader("저장된 기록")
            for i, record in enumerate(st.session_state.history):
                # 버튼 레이블: 기록 이름 (평균 점수)
                label = f"#{i+1}: {record['timestamp']} ({record['avg_score']}점)"
                is_selected = st.session_state.selected_history_index == i
                
                # 버튼 클릭 시 해당 기록 인덱스 저장 후 리런
                if st.button(label, key=f"hist_{i}", type=("primary" if is_selected else "secondary")):
                    st.session_state.selected_history_index = i
                    st.rerun()
        else:
            st.info("저장된 기록이 없습니다. 녹화 후 '녹화 중지'를 눌러주세요.")


    # -------------------------------
    # 메인 화면: 버튼 및 메트릭 표시
    # -------------------------------
    col1, col2 = st.columns([1, 4])
    
    # 버튼 로직
    if st.session_state.is_running:
        stop_button = col1.button("🛑 녹화 중지", key="stop_main", type="secondary")
        if stop_button:
            st.session_state.is_running = False
            
            # **녹화 중지 시, 최종 결과 저장 로직**
            if st.session_state.all_scores:
                avg_score = round(sum(st.session_state.all_scores)/len(st.session_state.all_scores), 2)
                
                st.session_state.history.append({
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.recording_start_time)),
                    'avg_score': avg_score,
                    'scores': st.session_state.all_scores,
                    'states': st.session_state.all_states,
                    'segments': len(st.session_state.all_scores)
                })
                # 저장 후, 저장된 기록을 보여주도록 selected_history_index를 마지막 기록으로 설정
                st.session_state.selected_history_index = len(st.session_state.history) - 1
                
            st.session_state.recording_start_time = 0 
            st.rerun()
            
        col2.markdown("## <span style='color:red;'>🔴 Recording...</span>", unsafe_allow_html=True)
        
    else:
        start_button = col1.button("▶️ 녹화 시작", key="start_main", type="primary")
        if start_button:
            st.session_state.is_running = True
            st.session_state.all_scores = []
            st.session_state.all_states = []
            st.session_state.segment_start = time.time()
            st.session_state.recording_start_time = time.time()
            # 결과 상태 초기화
            st.session_state.current_avg_score = 0
            st.session_state.current_stage_desc = "분석 시작 대기 중..."
            st.session_state.current_stage_level = 0
            st.session_state.current_total_time = "00:00:00"
            st.session_state.selected_history_index = None # 새 녹화 시작 시 선택 기록 해제
            st.rerun()
            
        col2.markdown("## ⚪ 대기 중")
        
    st.markdown("---")
    
    # -------------------------------
    # 결과 표시 영역 (메트릭/그래프)
    # -------------------------------

    if st.session_state.selected_history_index is not None:
        # **A. 저장된 기록 표시**
        record = st.session_state.history[st.session_state.selected_history_index]
        display_results(record['scores'], record['states'], record['segments'])
        
    elif st.session_state.is_running:
        # **B. 실시간 측정 중 표시**
        
        # 1. 메트릭 영역 표시
        elapsed_total_time = time.time() - st.session_state.recording_start_time
        hours = int(elapsed_total_time // 3600)
        minutes = int((elapsed_total_time % 3600) // 60)
        seconds = int(elapsed_total_time % 60)
        time_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        st_metrics = st.columns(3)
        st_metrics[0].metric("현재 집중도 (평균)", f"{st.session_state.current_avg_score} 점")
        st_metrics[1].metric("현재 집중 단계", f"단계 {st.session_state.current_stage_level}", st.session_state.current_stage_desc)
        # 총 녹화 시간을 여기서 업데이트 (10초마다만 움직이도록)
        st_time_text = st_metrics[2].metric("총 녹화 시간", st.session_state.current_total_time if st.session_state.current_total_time != "00:00:00" else time_display) 

        st_graph_area = st.empty() # 이 영역은 루프가 끝나기 전까지 비워둠
        st_penalty_area = st.empty()
        st_warning_area = st.empty() 

        # 2. Mediapipe 루프 실행
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st_warning_area.error("웹캠을 찾을 수 없거나 접근이 거부되었습니다. 측정을 시작할 수 없습니다.")
            st.session_state.is_running = False
            cap.release()
            return
            
        st.sidebar.info("측정 중... 웹캠이 백그라운드에서 활성화되었습니다.")
        
        blink_count = yawn_count = 0
        closed_seconds = half_closed_seconds = gaze_out_seconds = 0.0
        eye_closed = yawn_state = False
        
        with mp_holistic.Holistic(
            static_image_mode=False, model_complexity=1, refine_face_landmarks=True
        ) as holistic:

            while st.session_state.is_running:
                # ... (프레임 캡처 및 Mediapipe 분석 로직 유지)
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                frame_time = 1/30.0 

                if results.face_landmarks:
                    landmarks = results.face_landmarks.landmark
                    try:
                        left_eye = abs(landmarks[145].y - landmarks[159].y) * h
                        right_eye = abs(landmarks[374].y - landmarks[386].y) * h
                        eye_avg = (left_eye + right_eye) / 2
                        lip_dist = abs(landmarks[13].y - landmarks[14].y) * h
                        left_center_x = (landmarks[33].x + landmarks[133].x)/2
                        right_center_x = (landmarks[362].x + landmarks[263].x)/2
                        eye_center_x = (left_center_x + right_center_x)/2
                    except IndexError:
                        eye_avg, lip_dist, eye_center_x = 100, 0, 0.5 

                    # 눈 감김/깜빡임 처리
                    if eye_avg < CLOSED_THRESHOLD:
                        closed_seconds += frame_time
                        if not eye_closed:
                            eye_closed = True
                            blink_count += 1
                    else:
                        eye_closed = False

                    if CLOSED_THRESHOLD <= eye_avg < HALF_CLOSED_THRESHOLD:
                        half_closed_seconds += frame_time
                        
                    # 하품 처리
                    if lip_dist > YAWN_THRESHOLD:
                        if not yawn_state:
                            yawn_state = True
                            yawn_count += 1
                    else:
                        yawn_state = False

                    # 시선 이탈 처리
                    if not (0.5 - GAZE_THRESHOLD <= eye_center_x <= 0.5 + GAZE_THRESHOLD):
                        gaze_out_seconds += frame_time
                
                # 10초 단위 집중도 예측 및 업데이트
                current_time = time.time()
                elapsed_segment_time = current_time - st.session_state.segment_start
                
                if elapsed_segment_time >= 10:
                    score = calculate_focus_score(yawn_count, blink_count, closed_seconds, half_closed_seconds, gaze_out_seconds)
                    score = np.clip(score, 0, 100)
                    
                    st.session_state.all_scores.append(score)
                    st.session_state.all_states.append({
                        'blink_count': blink_count, 'yawn_count': yawn_count, 
                        'closed_seconds': closed_seconds, 'half_closed_seconds': half_closed_seconds,
                        'gaze_out_seconds': gaze_out_seconds
                    })

                    current_avg = round(sum(st.session_state.all_scores[-3:])/min(3, len(st.session_state.all_scores)), 2)
                    stage_level, stage_desc = get_focus_stage(current_avg)
                    
                    st.session_state.current_avg_score = current_avg
                    st.session_state.current_stage_level = stage_level
                    st.session_state.current_stage_desc = stage_desc
                    
                    # 총 녹화 시간 업데이트 및 상태 저장
                    elapsed_total_time = current_time - st.session_state.recording_start_time
                    hours = int(elapsed_total_time // 3600)
                    minutes = int((elapsed_total_time % 3600) // 60)
                    seconds = int(elapsed_total_time % 60)
                    st.session_state.current_total_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    blink_count = yawn_count = 0
                    closed_seconds = half_closed_seconds = gaze_out_seconds = 0.0
                    st.session_state.segment_start = current_time
                    
                    st.rerun() 
                
                time.sleep(0.01)

            cap.release()
            st.sidebar.info("측정 종료됨. 최종 결과를 확인하세요.")
            st.rerun() 

    elif not st.session_state.is_running and st.session_state.all_scores and st.session_state.selected_history_index is None:
        # **C. 현재 세션 종료 후 최종 결과 표시**
        display_results(
            st.session_state.all_scores, 
            st.session_state.all_states, 
            len(st.session_state.all_scores), 
            is_final=True
        )
        
    elif not st.session_state.is_running and not st.session_state.all_scores and st.session_state.selected_history_index is None:
        # **D. 초기 상태**
        st.info("측정을 시작하려면 '녹화 시작' 버튼을 클릭하고 웹캠을 활성화해 주세요.")
        
if __name__ == "__main__":
    main()