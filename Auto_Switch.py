from keras.models import load_model
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import time
from collections import Counter
import subprocess
import sys

# 載入Keras訓練好的模型和分類標籤
model = load_model("/Users/linjunting/Downloads/converted_keras-2/keras_model.h5", compile=False)
class_names = [line.strip() for line in open("/Users/linjunting/Downloads/converted_keras-2/labels.txt", "r").readlines()]

# 初始化相關變量
face_detected = False
start_time_face_detected = None

frame_interval = 1  # 每隔1幀進行一次完整的分析
frame_count = 0

# 定義在圖像上繪製文本的函數
def putText(img, text, x, y, size=32, color=(0, 0, 0)):
    font_path = "/Users/linjunting/Downloads/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf"
    font = ImageFont.truetype(font_path, size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)

# 處理幀函數
def process_frame(frame, class_names, model):
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    classification_image = np.asarray(resized_image, dtype=np.float32)
    classification_image = (classification_image / 127.5) - 1
    classification_image = classification_image.reshape(1, 224, 224, 3)
    prediction = model.predict(classification_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# 初始化攝像頭
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 設置較低解析度以提升效能
low_res_width = 320
low_res_height = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, low_res_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, low_res_height)

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    frame = cv2.resize(frame, (540, 320))  # 縮小尺寸，避免尺寸過大導致效能不好
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 將鏡頭影像轉換成灰階
    faces = face_cascade.detectMultiScale(gray)  # 偵測人臉

    # 偵測到人臉
    if len(faces) > 0:
        if not face_detected:
            face_detected = True
            start_time_face_detected = time.time()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 標記人臉

        # 偵測到人臉超過3秒
        if face_detected and (time.time() - start_time_face_detected) > 1:
            try:
                # 執行 project.py 腳本
                result = subprocess.run(['python', 'project.py'], capture_output=True, text=True)
                # 捕捉 project.py 的退出碼
                if result.returncode == 'q':
                    print("project.py returned 'q'")
                    break
            except Exception as e:
                print("Error in start the service:", e)
    else:
        face_detected = False
        start_time_face_detected = None

    cv2.imshow('oxxostudio', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()