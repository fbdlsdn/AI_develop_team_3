import cv2
import time

def process_frame(frame):
    """
    이부분에 Mediapipe 넣어야한다.
    """
    return frame

def run_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    duration = 90 * 60
    start_time = time.time()

    while True:
        if time.time() - start_time > duration:
            print("집중도 측정 완료")
            break
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        processed = process_frame(frame) #실시간 프레임 처리 구간 
        
        cv2.imshow("실시간 웹캠", processed) #화면 출력
        if cv2.waitKey(1) & 0xFF == ord('q'): #일단강종버튼 넣었음
            break
    
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run_camera()
        