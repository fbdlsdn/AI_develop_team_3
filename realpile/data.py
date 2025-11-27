import os, sys
sys.path.append(os.pardir)
import pandas as pd
import cv2

DATA_ROOT_DIR = '.\DAiSEE\DataSet'
LABEL_ROOT_DIR = '.\DAiSEE\Labels'

def data_loader(data_root: str, label_root: str, type: str):
    #type = 불러올 파일

    err_cnt = 0
    succes_cnt = 0
    processed_data = []

    label = pd.read_csv(os.path.join(label_root, type+"Labels.csv"))
    label_map = label.set_index("ClipID").to_dict('index')

    for video in os.listdir(os.path.join(data_root, type)):
        for video_frame in os.listdir(os.path.join(data_root, type, video)):

            clipID = video_frame + ".avi"
            video_path = os.path.join(data_root, type, video, video_frame, clipID)
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

            # 안 열리는 파일 검출
            if not cap.isOpened():
                print(f"Error: {video_path} open 실패")
                err_cnt += 1
                continue

            succes_cnt += 1

            if clipID in label_map:
                engagement = label_map[clipID]["Engagement"]
                # 처리된 데이터 저장
                processed_data.append({
                    "clip_id": clipID,
                    "file_path": video_path,
                    "engagement": engagement
                })

    print(f"{err_cnt}개의 파일을 불러오는데 실패했습니다\n{succes_cnt}개의 파일을 불러오는데 성공했습니다")

    return processed_data
                


data_loader(DATA_ROOT_DIR, LABEL_ROOT_DIR, "Train")
        