import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import os
import keyboard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------
# 경로 설정
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "DAiSEE", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "focus_model.pkl")
WEIGHT_PATH = os.path.join(MODEL_DIR, "best_feature_weights.npy")

# -------------------------------
# 기본 가중치 (fallback)
# -------------------------------
w1, w2, w3, w4 = 0.3, 0.2, 0.3, 0.2

# -------------------------------
# 모델 로드
# -------------------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("📌 모델 로드 완료: focus_model.pkl")
    except:
        print("⚠ 모델 로드 실패 — 기본 가중치만 사용")
else:
    print("⚠ 모델 없음 — 기본 가중치만 사용")


# -------------------------------
# 가중치 로드(best_feature_weights.npy)
# -------------------------------
if os.path.exists(WEIGHT_PATH):
    try:
        w = np.load(WEIGHT_PATH)
        if len(w) == 4:
            w = np.where(w == 0, 1e-6, w)
            w = np.round(w / w.sum(), 2)
            w1, w2, w3, w4 = w
            print("📌 학습된 가중치 적용:", (w1, w2, w3, w4))
        else:
            print("⚠ 가중치 파일 형식 오류 — 기본 가중치 사용")
    except Exception as e:
        print("⚠ 가중치 로드 실패:", e)
else:
    print("⚠ best_feature_weights.npy 없음 — 기본 가중치 사용:", (w1, w2, w3, w4))


# -------------------------------
# Mediapipe 초기화
# -------------------------------
mp_holistic = mp.solutions.holistic

# -------------------------------
# 기준값
# -------------------------------
CLOSED_THRESHOLD = 3
HALF_CLOSED_THRESHOLD = 6
YAWN_THRESHOLD = 25
GAZE_THRESHOLD = 0.45
BLINK_MAX = 20


# -------------------------------
# 집중도 계산 (가중치 반영)
# -------------------------------
def calculate_focus(yawn, blink, closed_time, half_closed_time, gaze_out_time,
                    w1, w2, w3, w4):

    penalty = 100 * (
        w1 * yawn +
        w2 * (blink / BLINK_MAX) +
        w3 * (closed_time / 10) +
        w4 * (gaze_out_time / 10)
    )
    return max(0, 100 - penalty)


# -------------------------------
# 패널티 분석(그래프용)
# -------------------------------
def analyze_focus_reason(blink, yawn, closed, half_closed, gaze):

    penalties = {
        'blink': blink * 0.2,
        'yawn': yawn * 0.3,
        'closed': closed * 0.3,
        'half_closed': half_closed * 0.1,
        'gaze_out': gaze * 0.2
    }
    return max(penalties, key=penalties.get), penalties


# -------------------------------
# 웹캠 시작
# -------------------------------
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_time = 1 / fps

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


with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True
) as holistic:

    print("웹캠 측정 시작... (q = 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.face_landmarks:
            lm = results.face_landmarks.landmark

            # 눈 감김
            left_eye = abs(lm[145].y - lm[159].y) * h
            right_eye = abs(lm[374].y - lm[386].y) * h
            avg_eye = (left_eye + right_eye) / 2

            if avg_eye < CLOSED_THRESHOLD:
                closed_seconds += frame_time
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

            if CLOSED_THRESHOLD <= avg_eye < HALF_CLOSED_THRESHOLD:
                half_closed_seconds += frame_time

            # 하품 감지
            lip_dist = abs(lm[13].y - lm[14].y) * h
            if lip_dist > YAWN_THRESHOLD:
                if not yawn_state:
                    yawn_count += 1
                    yawn_state = True
            else:
                yawn_state = False

            # 시선 이탈
            left_c = (lm[33].x + lm[133].x)/2
            right_c = (lm[362].x + lm[263].x)/2
            center_x = (left_c + right_c)/2

            if not (0.5 - GAZE_THRESHOLD <= center_x <= 0.5 + GAZE_THRESHOLD):
                gaze_out_seconds += frame_time

        # ==================================================
        # 10초마다 집중도 계산(가중치 적용)
        # ==================================================
        if time.time() - segment_start >= 10:

            score = calculate_focus(
                yawn_count,
                blink_count,
                closed_seconds,
                half_closed_seconds,
                gaze_out_seconds,
                w1, w2, w3, w4
            )

            all_scores.append(score)

            all_states.append({
                "blink_count": blink_count,
                "yawn_count": yawn_count,
                "closed_seconds": closed_seconds,
                "half_closed_seconds": half_closed_seconds,
                "gaze_out_seconds": gaze_out_seconds
            })

            # 초기화
            blink_count = yawn_count = 0
            closed_seconds = half_closed_seconds = 0
            gaze_out_seconds = 0
            segment_start = time.time()

        if keyboard.is_pressed('q'):
            print("측정 종료")
            break

cap.release()
cv2.destroyAllWindows()


# -------------------------------
# 최종 선그래프 출력
# -------------------------------
if all_scores:

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(all_scores)+1), all_scores, marker='o', linestyle='-', linewidth=2)
    plt.title("시간 흐름에 따른 집중도 변화 (학습된 가중치 기반)")
    plt.xlabel("10초 단위 구간")
    plt.ylabel("집중도 점수")
    plt.ylim(0, 100)
    plt.grid()
    plt.tight_layout()
    plt.show()

else:
    print("⚠ 집중도 데이터 부족.")
