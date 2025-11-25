import cv2
import mediapipe as mp
import math

# -------------------------------
# Mediapipe 초기화
# -------------------------------
mp_face = mp.solutions.face_mesh

# -------------------------------
# 눈 깜박임 계산용 함수
# -------------------------------
def eye_aspect_ratio(landmarks, eye_idx):
    p1 = landmarks[eye_idx[1]]
    p2 = landmarks[eye_idx[5]]
    p3 = landmarks[eye_idx[2]]
    p4 = landmarks[eye_idx[4]]
    p5 = landmarks[eye_idx[0]]
    p6 = landmarks[eye_idx[3]]

    vertical1 = math.dist(p2, p4)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p6)

    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

# -------------------------------
# Mediapipe 좌표 인덱스
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14

# -------------------------------
# 상태 초기화
# -------------------------------
blink_count = 0
yawn_count = 0
eye_closed = False  # 눈 감김 상태
yawn_state = False  # 하품 상태

# -------------------------------
# 프레임 처리
# -------------------------------
def process_frame(frame, face_mesh):
    global blink_count, yawn_count, eye_closed, yawn_state

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame

    for face in results.multi_face_landmarks:
        # 얼굴 랜드마크 점 그리기 제거
        landmarks = [(lm.x * w, lm.y * h) for lm in face.landmark]

        # -------------------------------
        # 1) 눈 깜박임 (완전히 감았다 뜰 때)
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        # 눈 감김 상태 판단
        if ear < 0.18:  # 눈 완전히 감김 기준
            eye_closed = True
        else:
            if eye_closed:
                blink_count += 1
            eye_closed = False

        # -------------------------------
        # 2) 하품 감지
        lip_dist = abs(landmarks[UPPER_LIP][1] - landmarks[LOWER_LIP][1])
        if lip_dist > 25:
            if not yawn_state:
                yawn_count += 1
                yawn_state = True
        else:
            yawn_state = False

        # -------------------------------
        # 3) 시선 방향
        left_eye_center = landmarks[33]
        right_eye_center = landmarks[362]
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2

        if eye_center_x < w * 0.4:
            gaze = "Left"
        elif eye_center_x > w * 0.6:
            gaze = "Right"
        else:
            gaze = "Center"

        # -------------------------------
        # 실시간 상태 표시
        cv2.putText(frame, f"Blink: {blink_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawn: {yawn_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Gaze: {gaze}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# -------------------------------
# 웹캠 실행
# -------------------------------
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            processed = process_frame(frame, face_mesh)
            cv2.imshow("Concentration Tracker", processed)

            # q 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # 최종 통계 출력
    print("=== 최종 집중도 통계 ===")
    print(f"총 눈 깜박임 횟수: {blink_count}")
    print(f"총 하품 횟수: {yawn_count}")

if __name__ == "__main__":
    run_camera()
