import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# 경로 설정
# -------------------------------
BASE = r"C:\IHATE.ZEROSTAR_SSAGAJI\realpile\DAiSEE"
TRAIN_LABEL = BASE + r"\Labels\TrainLabels.csv"
VALID_LABEL = BASE + r"\Labels\ValidationLabels.csv"
TEST_LABEL  = BASE + r"\Labels\TestLabels.csv"
TRAIN_DIR   = BASE + r"\Dataset\Train"
VALID_DIR   = BASE + r"\Dataset\Validation"
TEST_DIR    = BASE + r"\Dataset\Test"

# -------------------------------
# Mediapipe 초기화
# -------------------------------
mp_holistic = mp.solutions.holistic

CLOSED_THRESHOLD = 3
HALF_CLOSED_THRESHOLD = 6
YAWN_THRESHOLD = 25
GAZE_THRESHOLD = 0.45
BLINK_MAX = 20

# -------------------------------
# 영상에서 특징 추출 함수
# -------------------------------
def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    blink_count = 0
    yawn_count = 0
    closed_seconds = 0
    half_closed_seconds = 0
    gaze_out_seconds = 0
    eye_closed = False
    yawn_state = False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.face_landmarks:
                landmarks = results.face_landmarks.landmark

                # 눈 깜빡임
                left_eye = abs(landmarks[145].y - landmarks[159].y) * h
                right_eye = abs(landmarks[374].y - landmarks[386].y) * h
                eye_avg = (left_eye + right_eye) / 2

                if eye_avg < CLOSED_THRESHOLD:
                    closed_seconds += 1/fps
                    if not eye_closed:
                        eye_closed = True
                        blink_count += 1
                else:
                    eye_closed = False

                if CLOSED_THRESHOLD <= eye_avg < HALF_CLOSED_THRESHOLD:
                    half_closed_seconds += 1/fps

                # 하품
                lip_dist = abs(landmarks[13].y - landmarks[14].y) * h
                if lip_dist > YAWN_THRESHOLD:
                    if not yawn_state:
                        yawn_state = True
                        yawn_count += 1
                else:
                    yawn_state = False

                # 시선 이탈
                left_center = (landmarks[33].x + landmarks[133].x)/2
                right_center = (landmarks[362].x + landmarks[263].x)/2
                eye_center_x = (left_center + right_center)/2
                if not (0.5 - GAZE_THRESHOLD <= eye_center_x <= 0.5 + GAZE_THRESHOLD):
                    gaze_out_seconds += 1/fps

    cap.release()
    return [blink_count, yawn_count, closed_seconds, half_closed_seconds, gaze_out_seconds]

# -------------------------------
# 데이터셋 로드
# -------------------------------
def build_feature_dataset(label_file, root_dir):
    labels = pd.read_csv(label_file)
    label_map = labels.set_index("ClipID")['Engagement'].to_dict()

    X, y = [], []
    for video_name in os.listdir(root_dir):
        video_path = os.path.join(root_dir, video_name)
        if not os.path.isdir(video_path):
            continue
        clip_id = video_name + ".avi"
        features = extract_features_from_video(video_path)
        X.append(features)
        y.append(label_map.get(clip_id, 0))
    return np.array(X), np.array(y)

# -------------------------------
# 학습/검증/테스트 데이터 준비
# -------------------------------
print("특징 추출 중... (영상 길이에 따라 오래 걸릴 수 있음)")
X_train, y_train = build_feature_dataset(TRAIN_LABEL, TRAIN_DIR)
X_val, y_val     = build_feature_dataset(VALID_LABEL, VALID_DIR)
X_test, y_test   = build_feature_dataset(TEST_LABEL, TEST_DIR)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# -------------------------------
# MLPClassifier 학습
# -------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(32,8),
    activation='relu',
    solver='adam',
    learning_rate_init=0.0008,
    alpha=0.01,
    batch_size=32,
    max_iter=1,
    warm_start=True
)

best_val_loss = float('inf')
patience = 7
patience_counter = 0
train_losses, val_losses = [], []
best_model_path = BASE + r"\models\focus_model.pkl"
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

print("\n=== Training Start ===")
for epoch in range(1, 101):
    model.fit(X_train, y_train)
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"[Epoch {epoch:03}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        joblib.dump(model, best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early Stopping 발동 → Epoch {epoch}에서 종료")
            break

# -------------------------------
# 테스트셋 평가
# -------------------------------
model = joblib.load(best_model_path)
y_pred_test = model.predict(X_test)
print("\n=== Final Test Evaluation ===")
print(f"테스트 정확도: {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
print(classification_report(y_test, y_pred_test))

# -------------------------------
# 학습 곡선 시각화
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Learning Curve (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
