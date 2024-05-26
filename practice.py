from keras.models import load_model  # 從keras庫中導入加載模型的功能
import cv2  # 導入OpenCV庫用於影像處理
import numpy as np  # 導入numpy庫，用於數據處理
from deepface import DeepFace  # 從deepface庫導入DeepFace，用於臉部特徵分析
from PIL import ImageFont, ImageDraw, Image  # 從PIL庫導入圖像處理功能
import matplotlib.pyplot as plt  # 導入matplotlib庫的pyplot模塊，用於繪圖
import time  # 導入time庫，用於處理時間相關的功能

# 載入Keras訓練好的模型和分類標籤
model = load_model("/Users/linjunting/Downloads/converted_keras-2/keras_model.h5", compile=False)
class_names = [line.strip() for line in open("/Users/linjunting/Downloads/converted_keras-2/labels.txt", "r").readlines()]

# 初始化相關變量
class_1_detected = False  # 標記是否檢測到第一類
class_2_detected = False  # 標記是否檢測到第二類
start_time_1 = None  # 第一類檢測開始時間
start_time_2 = None  # 第二類檢測開始時間
start_time_low_confidence = None  # 信心水平低於100%的開始時間
emotions_over_time = []  # 紀錄時間序列中的情緒



class_1_detected1 = False  # 標記是否檢測到第一類
class_2_detected1 = False  # 標記是否檢測到第二類
start_time_11 = None  # 第一類檢測開始時間
start_time_21 = None  # 第二類檢測開始時間
start_time_low_confidence1 = None  # 信心水平低於100%的開始時間
emotions_over_time1 = []  # 紀錄時間序列中的情緒



# 定義情緒類別
emotion_categories = {
    'positive': ['happy', 'surprise'],  # 正面情緒
    'negative': ['angry', 'disgust', 'fear', 'sad'],  # 負面情緒
    'neutral': ['neutral']  # 中性情緒
}

# 定義在圖像上繪製文本的函數
def putText(img, text, x, y, size=32, color=(0, 0, 0)):
    font_path = "/Users/linjunting/Downloads/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf"
    font = ImageFont.truetype(font_path, size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)

# 啟動攝像頭
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")  # 無法開啟攝像頭
    exit()
if not cap1.isOpened():
    print("Cannot open camera")  # 無法開啟攝像頭
    exit()

# 主循環，不斷讀取攝像頭畫面
while True:
    ret, frame = cap.read() # 攝影機
    ret1, frame1 = cap1.read()
    if not ret:
        print("Cannot receive frame")  # 無法接收畫面
        break
    if not ret1:
        print("Cannot receive frame")
        break
    img0 = cv2.flip(cv2.resize(frame, (768, 480)), 1)
    img1 = cv2.flip(cv2.resize(frame1, (768, 480)), 1)

    # 對圖像進行預處理以符合模型輸入需求
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    resized_image1 = cv2.resize(frame1, (224, 224), interpolation=cv2.INTER_AREA)
    classification_image = np.asarray(resized_image, dtype=np.float32)
    classification_image1 = np.asarray(resized_image1, dtype=np.float32)
    classification_image = (classification_image / 127.5) - 1
    classification_image1 = (classification_image1 / 127.5) - 1
    classification_image = classification_image.reshape(1, 224, 224, 3)
    classification_image1 = classification_image1.reshape(1, 224, 224, 3)
    # 使用模型進行預測
    prediction = model.predict(classification_image)
    prediction1 = model.predict(classification_image1)
    index = np.argmax(prediction)
    index1 = np.argmax(prediction1)
    class_name = class_names[index]
    class_name1 = class_names[index1]
    confidence_score = prediction[0][index]
    confidence_score1 = prediction1[0][index1]

    # 處理類別1的檢測
    if class_name == 'Class 1':
        if confidence_score < 1:
            if not start_time_low_confidence:
                start_time_low_confidence = time.time()
            elif (time.time() - start_time_low_confidence) > 3:
                print("Class 1 confidence less than 100% for over 3 seconds, stopping emotion analysis.")
                break
        else:
            start_time_low_confidence = None  # 重置低信心水平時間
        if not class_1_detected:
            class_1_detected = True
            start_time_1 = time.time()
        class_2_detected = False
    elif class_name == 'Class 2':
        if not class_2_detected:
            class_2_detected = True
            start_time_2 = time.time()
        class_1_detected = False
    else:
        class_1_detected = False
        class_2_detected = False
        start_time_1 = None
        start_time_2 = None

    # 在檢測到類別1後3秒開始進行情緒分析
    if class_1_detected and start_time_1 and (time.time() - start_time_1) > 3:
        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analyze[0]['dominant_emotion']
            emotions_over_time.append(emotion)
            img0 = putText(img0, f"{class_name}, Confidence: {np.round(confidence_score * 100, 2)}%", 10, 30)
            img0 = putText(img0, f"Emotion: {emotion}", 10, 70)
        except Exception as e:
            print("Error in emotion detection:", e)
            
        try:
            analyze = DeepFace.analyze(frame1, actions=['emotion'], enforce_detection=False)
            emotion = analyze[0]['dominant_emotion']
            emotions_over_time1.append(emotion)
            img1 = putText(img1, f"{class_name1}, Confidence: {np.round(confidence_score1 * 100, 2)}%", 10, 30)
            img1 = putText(img1, f"Emotion: {emotion}", 10, 70)
        except Exception as e1:
                print("Error in emotion detection:", e1)        

    # 如果類別2連續檢測超過3秒則退出循環
    if class_2_detected and start_time_2 and (time.time() - start_time_2) > 3:
        print("Class 2 detected for more than 3 seconds, stopping emotion analysis.")
        break

    # 顯示攝像頭圖像
    cv2.imshow('camera0', img0)
    cv2.imshow('camera1', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭和關閉所有視窗
cap.release()
cap1.release()
cv2.destroyAllWindows()

#各情緒權重
basicpoint=60
remain=100-basicpoint
negativeweight=-1
neutralweight=0
positiveweight=1

# 鏡頭一情緒轉換成1,0,-1
emotions_mapped0 = []
negative0=0
neutral0=0
positive0=0
for e in emotions_over_time:
    if e in emotion_categories['positive']:
        emotions_mapped0.append(1)
        positive0+=1
    elif e in emotion_categories['negative']:
        emotions_mapped0.append(-1)
        negative0+=1
    elif e in emotion_categories['neutral']:
        emotions_mapped0.append(0)
        neutral0+=1

#算鏡頭一各情緒百分比
total_emotions0 = negative0 + neutral0 + positive0
if total_emotions0 > 0:
    negative0perc = round(negative0 / total_emotions0,2)
    neutral0perc = round(neutral0 / total_emotions0,2)
    positive0perc = round(positive0 / total_emotions0,2)
else:
    negative0perc = neutral0perc = positive0perc = 0

#算鏡頭一分析後分數
cam0scr=basicpoint+negative0perc*remain*negativeweight+neutral0perc*remain*neutralweight+positive0perc*remain*positiveweight


#鏡頭二情緒轉換成1,0,-1
emotions_mapped1 = []
negative1=0
neutral1=0
positive1=0
for e in emotions_over_time1:
    if e in emotion_categories['positive']:
        emotions_mapped1.append(1)
        positive1+=1
    elif e in emotion_categories['negative']:
        emotions_mapped1.append(-1)
        negative1+=1
    elif e in emotion_categories['neutral']:
        emotions_mapped1.append(0)
        neutral1+=1

#算鏡頭二各情緒百分比
total_emotions1 = negative1 + neutral1 + positive1
if total_emotions1 > 0:
    negative1perc = round(negative1 / total_emotions1,2)
    neutral1perc = round(neutral1 / total_emotions1,2)
    positive1perc = round(positive1 / total_emotions1,2)
else:
    negative1perc = neutral1perc = positive1perc = 0

#算鏡頭二分析後分數
cam1scr=basicpoint+negative1perc*remain*negativeweight+neutral1perc*remain*neutralweight+positive1perc*remain*positiveweight



# 以下動作將情緒映射為數值，並繪製情緒波動折線圖

#畫鏡頭一折線圖
emotions_mapped = [1 if e in emotion_categories['positive'] else -1 if e in emotion_categories['negative'] else 0 for e in emotions_over_time]
plt.figure(figsize=(10, 5))
plt.plot(emotions_mapped, label='Emotion Wave', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.title("Emotion Wave Over Time")
plt.xlabel("Frame")
plt.ylabel("Emotion")
plt.legend()
plt.show()


#畫鏡頭一長條圖
emotions = ['Negative', 'Neutral', 'Positive']
percentages0 = [negative0perc, neutral0perc, positive0perc]
plt.figure(figsize=(8, 4))
bars0=plt.bar(emotions, percentages0, color=['red', 'gray', 'green'])
plt.title('Percentage of Each Emotion in Cam0 - Full Survice Score: {cam0scr:.2f}')
plt.xlabel('Emotion')
plt.ylabel('Percentage')
plt.ylim(0, 1)
for bar in bars0:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=10, color='black')


#畫鏡頭二折線圖
emotions_mapped = [1 if e1 in emotion_categories['positive'] else -1 if e1 in emotion_categories['negative'] else 0 for e1 in emotions_over_time1]
plt.figure(figsize=(10, 5))
plt.plot(emotions_mapped, label='Emotion Wave', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.title("Emotion Wave Over Time")
plt.xlabel("Frame")
plt.ylabel("Emotion")
plt.legend()
plt.show()

# 畫鏡頭二長條圖
percentages1 = [negative1perc, neutral1perc, positive1perc]
plt.figure(figsize=(8, 4))
bars1=plt.bar(emotions, percentages1, color=['red', 'gray', 'green'])
plt.title('Percentage of Each Emotion in Cam1 - Full Service Score: {cam1scr:.2f}')
plt.xlabel('Emotion')
plt.ylabel('Percentage')
plt.ylim(0, 1)
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=10, color='black')

plt.show()
