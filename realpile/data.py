import os
import sys
sys.path.append(os.pardir)
import pandas as pd
import cv2

# -------------------------------
# 절대경로 기준으로 수정
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # realpile 폴더
DATA_ROOT_DIR = os.path.join(BASE_DIR, "DAiSEE", "DataSet")
LABEL_ROOT_DIR = os.path.join(BASE_DIR, "DAiSEE", "Labels")

def data_loader(data_root: str, label_root: str, type: str):
    # type = 불러올 파일 (Train / Test / Validation 등)

    err_cnt = 0
    succes_cnt = 0
    processed_data = []

    # 파일 존재 여부 체크
    label_path = os.path.join(label_root, type + "Labels.csv")
    if not os.path.exists(label_path):
        print(f"Error: {label_path} 파일이 존재하지 않습니다.")
        return processed_data

    label = pd.read_csv(label_path)
    label_map = label.set_index("ClipID").to_dict('index')

    type_dir = os.path.join(data_root, type)
    if not os.path.exists(type_dir):
        print(f"Error: {type_dir} 디렉토리가 존재하지 않습니다.")
        return processed_data

    for video in os.listdir(type_dir):
        video_path = os.path.join(type_dir, video)
        if not os.path.isdir(video_path):
            continue

        for video_frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, video_frame)
            if not os.path.isdir(frame_path):
                continue

            clipID = video_frame + ".avi"
            video_file_path = os.path.join(frame_path, clipID)
            cap = cv2.VideoCapture(video_file_path, cv2.CAP_FFMPEG)

            # 안 열리는 파일 검출
            if not cap.isOpened():
                print(f"Error: {video_file_path} open 실패")
                err_cnt += 1
                continue

            succes_cnt += 1

            if clipID in label_map:
                engagement = label_map[clipID]["Engagement"]
                # 처리된 데이터 저장
                processed_data.append({
                    "clip_id": clipID,
                    "file_path": video_file_path,
                    "engagement": engagement
                })

    print(f"{err_cnt}개의 파일을 불러오는데 실패했습니다\n{succes_cnt}개의 파일을 불러오는데 성공했습니다")

    return processed_data

# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    # 파일 존재 여부 출력 (디버깅용)
    train_label_path = os.path.join(LABEL_ROOT_DIR, "TrainLabels.csv")
    print("TrainLabels.csv 경로:", train_label_path)
    print("파일 존재 여부:", os.path.exists(train_label_path))

    data_loader(DATA_ROOT_DIR, LABEL_ROOT_DIR, "Train")
