from keras.models import load_model
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt

# Load the Keras model
model = load_model("/Users/linjunting/Downloads/converted_keras/keras_model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("/Users/linjunting/Downloads/converted_keras/labels.txt", "r").readlines()]

# 定義情緒類別
emotion_categories = {
    'positive': ['happy', 'surprise'],
    'negative': ['angry', 'disgust', 'fear', 'sad'],
    'neutral': ['neutral']
}

# 紀錄每一幀情緒
emotions_over_time = []

# 定義加入文字函數
def putText(img, x, y, text, size=32, color=(255, 255, 255)):
    fontpath = '/Users/linjunting/Downloads/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf'  
    font = ImageFont.truetype(fontpath, size)
    imgPil = Image.fromarray(img)
    draw = ImageDraw.Draw(imgPil)
    draw.text((x, y), text, fill=color, font=font)
    return np.array(imgPil)

# 初始拍攝鏡頭
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    # Resize the image for the classification model
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Prepare the image for classification
    classification_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    classification_image = (classification_image / 127.5) - 1
    
    # Predict the class
    prediction = model.predict(classification_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    try:
        # 進行情緒分析
        analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analyze[0]['dominant_emotion']

        # 紀錄情绪
        emotions_over_time.append(emotion)

        # 在圖上顯示分類结果和情绪
        frame = putText(frame, 10, 30, f"{class_name}, Confidence: {np.round(confidence_score * 100, 2)}%", 30, (255, 255, 0))
        frame = putText(frame, 10, 70, f"Emotion: {emotion}", 30, (255, 255, 0))
    except Exception as e:
        print(e)

    cv2.imshow('Webcam Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# 把情绪轉為数值（正面為1，負面為-1，neutral為0）
emotions_mapped = []
for e in emotions_over_time:
    if e in emotion_categories['positive']:
        emotions_mapped.append(1)
    elif e in emotion_categories['negative']:
        emotions_mapped.append(-1)
    elif e in emotion_categories['neutral']:
        emotions_mapped.append(0)

# 畫出波浪圖
plt.figure(figsize=(10, 5))
plt.plot(emotions_mapped, label='Emotion Wave', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')  # 添加neutral情绪的参考線
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.title("Emotion Wave Over Time")
plt.xlabel("Frame")
plt.ylabel("Emotion")
plt.legend()
plt.show()
