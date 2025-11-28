import cv2
import mediapipe as mp
import math

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

# 눈 감김 민감도 및 시선 이동 민감도
CLOSED_THRESHOLD = 2           # 눈 높이(px) 기준
GAZE_MOVE_THRESHOLD = 5         # 픽셀 이동 기준
YAWN_THRESHOLD = 25             # 입 벌림(px) 기준

# -------------------------------
# 프레임 처리 함수
# -------------------------------
def process_frame(frame, holistic):
    global blink_count, eye_closed, yawn_count, yawn_state
    global prev_eye_center, gaze_change_count

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    if results.face_landmarks:
        landmarks = results.face_landmarks.landmark

        # -------------------------------
        # 1) 눈 깜박임 감지 (좌/우 눈 중심 거리 기준)
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        left_eye_height = abs(left_eye_bottom.y - left_eye_top.y) * h
        right_eye_height = abs(right_eye_bottom.y - right_eye_top.y) * h
        eye_avg_height = (left_eye_height + right_eye_height) / 2

        if eye_avg_height < CLOSED_THRESHOLD:
            if not eye_closed:
                eye_closed = True
                blink_count += 1
        else:
            eye_closed = False

        # -------------------------------
        # 2) 하품 감지 (입 벌림 거리)
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        lip_dist = abs(upper_lip.y - lower_lip.y) * h

        if lip_dist > YAWN_THRESHOLD:
            if not yawn_state:
                yawn_count += 1
                yawn_state = True
        else:
            yawn_state = False

        # -------------------------------
        # 3) 시선 변화 감지
        left_eye_center = ((landmarks[33].x + landmarks[133].x)/2 * w,
                           (landmarks[33].y + landmarks[133].y)/2 * h)
        right_eye_center = ((landmarks[362].x + landmarks[263].x)/2 * w,
                            (landmarks[362].y + landmarks[263].y)/2 * h)

        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
        eye_center = (eye_center_x, eye_center_y)

        if prev_eye_center is not None:
            move_dist = math.dist(prev_eye_center, eye_center)
            if move_dist > GAZE_MOVE_THRESHOLD:
                gaze_change_count += 1
        prev_eye_center = eye_center

        # 시선 방향 표시
        if eye_center_x < w*0.4:
            gaze = "Left"
        elif eye_center_x > w*0.6:
            gaze = "Right"
        else:
            gaze = "Center"

        # -------------------------------
        # 화면 표시
        cv2.putText(frame, f"Blink: {blink_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawn: {yawn_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Gaze: {gaze}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze Change: {gaze_change_count}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

    return frame

# -------------------------------
# 웹캠 실행
# -------------------------------
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame, holistic)
            cv2.imshow("Holistic Blink & Gaze Tracker", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("총 눈 깜박임:", blink_count)
    print("총 하품:", yawn_count)
    print("총 시선 변화:", gaze_change_count)

# -------------------------------
# 메인 실행
# -------------------------------
if __name__ == "__main__":
    run_camera()
