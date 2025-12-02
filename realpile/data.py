import os
import pandas as pd
import cv2

# -------------------------------
# 절대경로 기준으로 수정
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # realpile 폴더
DATA_ROOT_DIR = os.path.join(BASE_DIR, "DAiSEE", "Dataset")
LABEL_ROOT_DIR = os.path.join(BASE_DIR, "DAiSEE", "Labels")

def data_loader(type: str):
    """
    type: "Train", "Validation", "Test"
    """
    processed_data = []
    err_cnt = 0
    succes_cnt = 0

    # 라벨 파일 경로
    label_path = os.path.join(LABEL_ROOT_DIR, f"{type}Labels.csv")
    if not os.path.exists(label_path):
        print(f"Error: {label_path} 파일이 존재하지 않습니다.")
        return processed_data

    label_df = pd.read_csv(label_path)
    label_map = label_df.set_index("ClipID").to_dict("index")

    type_dir = os.path.join(DATA_ROOT_DIR, type)
    if not os.path.exists(type_dir):
        print(f"Error: {type_dir} 디렉토리가 존재하지 않습니다.")
        return processed_data

    for video in os.listdir(type_dir):
        video_path = os.path.join(type_dir, video)
        if not os.path.isdir(video_path):
            continue

        for frame_folder in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame_folder)
            if not os.path.isdir(frame_path):
                continue

            clipID = frame_folder + ".avi"
            video_file_path = os.path.join(frame_path, clipID)
            cap = cv2.VideoCapture(video_file_path, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print(f"Error: {video_file_path} open 실패")
                err_cnt += 1
                continue

            succes_cnt += 1

            if clipID in label_map:
                engagement = label_map[clipID]["Engagement"]
                processed_data.append({
                    "clip_id": clipID,
                    "file_path": video_file_path,
                    "engagement": engagement
                })

    print(f"{err_cnt}개의 파일 불러오기 실패, {succes_cnt}개의 파일 불러오기 성공")
    return processed_data

# -------------------------------
# 단독 실행 시 확인
# -------------------------------
if __name__ == "__main__":
    train_data = data_loader("Train")
    print(f"Train 데이터 수: {len(train_data)}")
