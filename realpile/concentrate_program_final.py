import cv2
import mediapipe as mp
import math
import time

# -------------------------------
# Mediapipe Holistic 초기화
# -------------------------------
mp_holistic = mp.solutions.holistic

# -------------------------------
# 상태 초기화
# -------------------------------
blink_count = 0
yawn_count = 0
gaze_change_count = 0
eye_closed = False
yawn_state = False
prev_eye_center = None
closed_seconds = 0   # 눈 감은 시간(초)

# 민감도 설정
CLOSED_THRESHOLD = 5           # 눈 높이(px) 기준
GAZE_MOVE_THRESHOLD = 5        # 픽셀 이동 기준
YAWN_THRESHOLD = 25            # 입 벌림(px) 기준

# -------------------------------
# 집중도 계산 함수 (가중치 적용)
# -------------------------------
def calculate_focus(yawn, blink, closed_time, w1=0.3, w2=0.2, w3=0.5, blink_max=20):
    """
    집중도 수식:
    S(t) = max(0, 100 - (w1*Nyawn(t) + w2*(Nblink(t)/Nblink_max) + w3*(Tclosed(t)/10)))
    """
    score = 100 - (w1*yawn + w2*(blink/blink_max) + w3*(closed_time/10))
    return max(0, score)

# -------------------------------
# 프레임 처리 함수
# -------------------------------
def process_frame(frame, holistic):
    global blink_count, eye_closed, yawn_count, yawn_state
    global prev_eye_center, gaze_change_count, closed_seconds

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    if results.face_landmarks:
        landmarks = results.face_landmarks.landmark

        # 1) 눈 깜박임 감지
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        left_eye_height = abs(left_eye_bottom.y - left_eye_top.y) * h
        right_eye_height = abs(right_eye_bottom.y - right_eye_top.y) * h
        eye_avg_height = (left_eye_height + right_eye_height) / 2

        if eye_avg_height < CLOSED_THRESHOLD:
            closed_seconds += 1/30.0  # 대략 30fp산
            if time.time() - start_segment >= 10:
                score = calculate_focus(yawn_count, blink_count, closed_seconds)
                segment_scores.append(score)
                print(f"{len(segment_scores)}번째 10초 집중도: {score:.2f}")

                # 카운트 초기화
                blink_count = 0
                yawn_count = 0
                gaze_change_count = 0
                closed_seconds = 0
                start_segment = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):  # q 키로 종료
                break

    cap.release()
    cv2.destroyAllWindows()

    # 최종 집중도 통계 출력
    if segment_scores:
        avg_focus = sum(segment_scores) / len(segment_scores)
        best_segment = segment_scores.index(max(segment_scores)) + 1
        print("=== 최종 집중도 통계 ===")
        print("평균 집중도:", avg_focus)
        print("최고 집중도 구간:", best_segment, "번째 10초")
    else:
        print("집중도 데이터가 없습니다.")

# -------------------------------
# 메인 실행
# -------------------------------
if __name__ == "__main__":
    run_camera()
