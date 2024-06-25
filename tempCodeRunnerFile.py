import cv2
import time

# 初始化兩個攝像頭
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# 設置幀率和影像尺寸
target_fps = 10
cap0.set(cv2.CAP_PROP_FPS, target_fps)
cap1.set(cv2.CAP_PROP_FPS, target_fps)

width0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
height0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定影片編碼和創建VideoWriter物件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out0 = cv2.VideoWriter('output_cam0.mp4', fourcc, target_fps, (width0, height0))
out1 = cv2.VideoWriter('output_cam1.mp4', fourcc, target_fps, (width1, height1))

# 確認攝像頭是否打開
if not cap0.isOpened() or not cap1.isOpened():
    print("Cannot open camera")
    exit()

frame_interval = 1 / target_fps  # 計算每幀間的延遲時間

while True:
    start_time = time.time()
    
    # 讀取兩個攝像頭的幀
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1:
        print("Cannot receive frame")
        break

    # 寫入幀到影片文件
    out0.write(frame0)
    out1.write(frame1)

    # 顯示兩個攝像頭的幀
    cv2.imshow('Camera 0', frame0)
    cv2.imshow('Camera 1', frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 計算處理幀所需的時間，並保持幀速率穩定
    elapsed_time = time.time() - start_time
    time_to_wait = max(0, frame_interval - elapsed_time)
    time.sleep(time_to_wait)

# 釋放攝像頭和VideoWriter物件
cap0.release()
cap1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()
