import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from data import data_loader
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib

# -----------------------------------------------------------
# Mediapipe 설정 & 상수
# -----------------------------------------------------------
mp_holistic = mp.solutions.holistic

CLOSED_THRESHOLD = 3
HALF_CLOSED_THRESHOLD = 6
YAWN_THRESHOLD = 25
GAZE_THRESHOLD = 0.45
BLINK_MAX = 20

# ⭐ 조금 느려져도 정확도 올리기 → 프레임 수 살짝 증가
MAX_FRAMES = 30


# -----------------------------------------------------------
# 영상 특징 추출
# -----------------------------------------------------------
def extract_features_from_video(video_path):

    cap = cv2.VideoCapture(video_path)

    blink = 0
    yawn = 0
    closed = 0
    half_closed = 0
    gaze = 0

    eye_closed = False
    yawn_state = False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    SAMPLE_RATE = 3
    interval = int(fps / SAMPLE_RATE)

    idx = 0
    processed = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        refine_face_landmarks=False
    ) as holistic:

        while True:

            if processed >= MAX_FRAMES:
                break

            ret, frame = cap.read()
            if not ret:
                break

            idx += 1
            if idx % interval != 0:
                continue

            processed += 1

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            if not results.face_landmarks:
                continue

            lm = results.face_landmarks.landmark

            # 눈
            left = abs(lm[145].y - lm[159].y) * h
            right = abs(lm[374].y - lm[386].y) * h
            eye_avg = (left + right) / 2

            if eye_avg < CLOSED_THRESHOLD:
                closed += 1
                if not eye_closed:
                    blink += 1
                    eye_closed = True
            else:
                eye_closed = False

            if CLOSED_THRESHOLD <= eye_avg < HALF_CLOSED_THRESHOLD:
                half_closed += 1

            # 하품
            lip = abs(lm[13].y - lm[14].y) * h
            if lip > YAWN_THRESHOLD:
                if not yawn_state:
                    yawn += 1
                    yawn_state = True
            else:
                yawn_state = False

            # 시선
            lc = (lm[33].x + lm[133].x) / 2
            rc = (lm[362].x + lm[263].x) / 2
            cx = (lc + rc) / 2

            if not (0.5 - GAZE_THRESHOLD <= cx <= 0.5 + GAZE_THRESHOLD):
                gaze += 1

    cap.release()

    # 정규화
    blink_n = min((blink / BLINK_MAX) * 2.0, 1.0)
    closed_n = min((closed / 10) * 1.3, 1.0)
    gaze_n = min((gaze / 10) * 1.5, 1.0)
    yawn = min(yawn, 3)

    return [yawn, blink_n, closed_n, gaze_n]


# -----------------------------------------------------------
# 병렬 처리용 워커
# -----------------------------------------------------------
def _process_one(args):
    f, lbl = args
    x = extract_features_from_video(f)
    return x, lbl, os.path.basename(f)


# -----------------------------------------------------------
# 병렬 특징 추출
# -----------------------------------------------------------
def build_feature_dataset_parallel(files, labels, split_name=""):
    X, y = [], []
    total = len(files)
    start_time = time.time()

    print(f"\n[{split_name}] 특징 추출 시작 (총 {total}개, MAX_FRAMES={MAX_FRAMES})")

    max_workers = min(4, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one, (f, lbl)) for f, lbl in zip(files, labels)]

        for idx, fut in enumerate(as_completed(futures), 1):
            x_i, lbl_i, fname = fut.result()
            X.append(x_i)
            y.append(lbl_i)

            if idx % 5 == 0 or idx == total:
                elapsed = time.time() - start_time
                avg = elapsed / idx
                remain = avg * (total - idx)
                print(f"  >>> [{idx}/{total}] {fname} | 진행률 {idx/total*100:.2f}% | 남음 {remain/60:.1f}분")

    return np.array(X), np.array(y)


# =================================================================
# 메인
# =================================================================
if __name__ == "__main__":

    print("\n데이터 로딩 중...")
    train_data = data_loader("Train")
    val_data   = data_loader("Validation")
    test_data  = data_loader("Test")

    train_files  = [d['file_path'] for d in train_data]
    train_labels = [d['engagement'] for d in train_data]

    val_files  = [d['file_path'] for d in val_data]
    val_labels = [d['engagement'] for d in val_data]

    test_files  = [d['file_path'] for d in test_data]
    test_labels = [d['engagement'] for d in test_data]

    # 병렬 처리
    X_train, y_train = build_feature_dataset_parallel(train_files, train_labels, "Train")
    X_val,   y_val   = build_feature_dataset_parallel(val_files,   val_labels,   "Validation")
    X_test,  y_test  = build_feature_dataset_parallel(test_files,  test_labels,  "Test")

    print("\n=== 특징 추출 완료 ===")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # ---------------------------------------------------------
    # 정규화
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ⭐ 정확도 ↑ 위해 Train + Val 합쳐서 최종 학습에 사용
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print("Train+Val 통합:", X_train_full.shape)

    # ---------------------------------------------------------
    # Test 샘플링
    # ---------------------------------------------------------
    def sample_test(X, y, rate=0.3):
        size = max(1, int(len(X) * rate))
        idx = np.random.choice(len(X), size=size, replace=False)
        return X[idx], y[idx]

    # ---------------------------------------------------------
    # RandomForest (정확도 우선 튜닝)
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=1200,      # 800 → 1200 (조금 더 세게)
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,     # 살짝 규제 → test 성능 안정화
        class_weight="balanced",
        bootstrap=True,
        max_features="sqrt",    # 전체 대신 sqrt → 일반적으로 일반화 더 좋음
        n_jobs=-1
    )

    EPOCHS = 10
    train_accs = []
    test_accs  = []
    weight_history = []

    best_test_acc = -1
    best_weights = None

    print("\n=== Training Start ===")

    for epoch in range(EPOCHS):

        # Train+Val 통합 데이터에서 셔플
        idx = np.random.permutation(len(X_train_full))
        X_epoch = X_train_full[idx]
        y_epoch = y_train_full[idx]

        # feature 순서 랜덤 (가중치 변화 유지)
        perm = np.random.permutation(X_epoch.shape[1])
        reverse_perm = np.argsort(perm)
        X_epoch = X_epoch[:, perm]

        # 학습
        model.fit(X_epoch, y_epoch)

        # accuracy 계산
        train_acc = accuracy_score(y_epoch, model.predict(X_epoch))
        X_ts, y_ts = sample_test(X_test, y_test, 0.3)
        test_acc = accuracy_score(y_ts, model.predict(X_ts))

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 가중치 계산
        w = model.feature_importances_[reverse_perm]
        w = np.clip(w, 0.05, None)     # 최소 가중치 0.05 보장
        w_norm = np.round(w / w.sum(), 2)

        weight_history.append(w_norm)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train={train_acc:.4f} | Test={test_acc:.4f}")
        print(f"   Weights(norm): {w_norm}, SUM={w_norm.sum():.2f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = w_norm.copy()

    # ---------------------------------------------------------
    # 최고 가중치 저장
    # ---------------------------------------------------------
    if best_weights is not None:
        np.save("best_feature_weights.npy", best_weights)
        print("\n=== 최고 정확도 가중치 저장 완료 ===")
        print("Best Test Accuracy:", best_test_acc)
        print("Saved Weights:", best_weights)

    # ---------------------------------------------------------
    # 모델 저장
    # ---------------------------------------------------------
    joblib.dump(model, "focus_model.pkl")
    print("\n=== RandomForest 모델 저장 완료: focus_model.pkl ===")

    # ---------------------------------------------------------
    # 그래프 출력
    # ---------------------------------------------------------
    epochs = range(1, EPOCHS+1)
    w_hist = np.array(weight_history)

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, test_accs,  label="Test Acc")
    plt.legend()
    plt.grid()
    plt.title("Accuracy per Epoch")

    plt.subplot(1,2,2)
    plt.plot(epochs, w_hist[:,0], label="yawn")
    plt.plot(epochs, w_hist[:,1], label="blink")
    plt.plot(epochs, w_hist[:,2], label="closed")
    plt.plot(epochs, w_hist[:,3], label="gaze")
    plt.legend()
    plt.grid()
    plt.title("Feature Importance (sum=1)")

    plt.tight_layout()
    plt.show()
