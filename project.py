from keras.models import load_model
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import time
from collections import Counter
import sys

# 載入Keras訓練好的模型和分類標籤
model = load_model("/Users/linjunting/Downloads/converted_keras-2/keras_model.h5", compile=False)
class_names = [line.strip() for line in open("/Users/linjunting/Downloads/converted_keras-2/labels.txt", "r").readlines()]

# 初始化相關變量
class_1_detected = False
class_2_detected = False
start_time_1 = None
start_time_2 = None
start_time_low_confidence = None
ages_over_time = []
genders_over_time = []
emotions_over_time = []

class_1_detected1 = False
class_2_detected1 = False
start_time_11 = None
start_time_21 = None
start_time_low_confidence1 = None
ages_over_time1 = []
genders_over_time1 = []
emotions_over_time1 = []
result_age = None
result1_age = None
result_gender = None
result1_gender = None
result_gender_confidence = None
result1_gender_confidence = None
check_time = 0

# 定義情緒類別
emotion_categories = {
    'positive': ['happy', 'surprise'],
    'negative': ['angry', 'sad'],
    'neutral': ['neutral', 'disgust', 'fear']
}

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


# 分析情緒、年齡和性別的函數
def analyze_frame_A(frame, class_name, confidence_score, emotions_over_time, ages_over_time, genders_over_time, check_time):
    try:
        analyze = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        emotion = analyze[0]['dominant_emotion']
        age = round(analyze[0]['age'])
        gender_prob = analyze[0]['gender']
        gender = max(gender_prob, key=gender_prob.get)
        gender_confidence = round(gender_prob[gender], 2)
        emotions_over_time.append(emotion)
        ages_over_time.append(age)
        genders_over_time.append((gender, gender_confidence))
        return {
            'class_name': class_name, 'confidence_score': np.round(confidence_score * 100, 2),
            'emotion': emotion, 'age': age, 'gender': gender, 'gender_confidence': gender_confidence
        }
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None

# 分析情緒的函數
def analyze_frame_B(frame, class_name, confidence_score, emotions_over_time, check_time):
    try:
        analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analyze[0]['dominant_emotion']
        emotions_over_time.append(emotion)
        return {
            'class_name': class_name, 'confidence_score': np.round(confidence_score * 100, 2),
            'emotion': emotion
        }
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None

# 啟動攝像頭
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)


# 設置幀率和影像尺寸
target_fps = 5
cap.set(cv2.CAP_PROP_FPS, target_fps)
cap1.set(cv2.CAP_PROP_FPS, target_fps)

# 設置較低解析度以提升效能
low_res_width = 320
low_res_height = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, low_res_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, low_res_height)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, low_res_width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, low_res_height)

width0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定影片編碼和創建VideoWriter物件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out0 = cv2.VideoWriter('output_cam0.mp4', fourcc, target_fps, (width0, height0))
out1 = cv2.VideoWriter('output_cam1.mp4', fourcc, target_fps, (width1, height1))


if not cap.isOpened() or not cap1.isOpened():
    print("Cannot open camera")
    exit()

frame_interval = 1  # 每隔1幀進行一次完整的分析
frame_count = 0

# 初始化保存結果的變量
previous_results = {
    'class_name': '', 'confidence_score': 0, 'emotion': '', 'age': 0,
    'gender': '', 'gender_confidence': 0,
    'class_name1': '', 'confidence_score1': 0, 'emotion1': '',
    'age1': 0, 'gender1': '', 'gender_confidence1': 0
}

# 主循環，不斷讀取攝像頭畫面
while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if not ret or not ret1:
        print("Cannot receive frame")
        break

    img0 = cv2.flip(cv2.resize(frame, (768, 480)), 1)
    img1 = cv2.flip(cv2.resize(frame1, (768, 480)), 1)

    if frame_count % frame_interval == 0:
        # 在主循環中調用
        class_name, confidence_score = process_frame(frame, class_names, model)
        class_name1, confidence_score1 = process_frame(frame1, class_names, model)

        # 處理類別1的檢測
        if class_name == 'Class 1':
            if confidence_score < 1:
                if not start_time_low_confidence:
                    start_time_low_confidence = time.time()
                elif (time.time() - start_time_low_confidence) > 3:
                    print("Class 1 confidence less than 100% for over 3 seconds, stopping emotion analysis.")
                    break
            else:
                start_time_low_confidence = None
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

        if class_1_detected and start_time_1 and (time.time() - start_time_1) > 3:
            check_time = time.time() - start_time_1
            if check_time <= 8: #超過分析超過五秒年齡性別得出結論->直接使用第五秒最後一次判斷的性別年齡結果
                result = analyze_frame_A(frame, class_name, confidence_score, emotions_over_time, ages_over_time, genders_over_time, check_time)
                result_age = result['age']
                result_gender = result['gender']
                result_gender_confidence = result['gender_confidence']
            else:
                result = analyze_frame_B(frame, class_name, confidence_score, emotions_over_time, check_time)
            if result:
                previous_results.update(result)
                img0 = putText(img0, f"{result['class_name']}, Confidence: {result['confidence_score']}%", 10, 30)
                img0 = putText(img0, f"Emotion: {result['emotion']}", 10, 70)
                img0 = putText(img0, f"Age: {result_age}", 10, 110)
                img0 = putText(img0, f"Gender: {result_gender} {result_gender_confidence}%", 10, 150)
                
            if check_time <= 8: #超過分析超過五秒年齡性別得出結論->直接使用第五秒最後一次判斷的性別年齡結果
                result1 = analyze_frame_A(frame1, class_name1, confidence_score1, emotions_over_time1, ages_over_time1, genders_over_time1, check_time)
                result1_age = result1['age']
                result1_gender = result1['gender']
                result1_gender_confidence = result1['gender_confidence']
            else:
                result1 = analyze_frame_B(frame1, class_name1, confidence_score1, emotions_over_time1, check_time)
            if result1:
                previous_results.update(result1)
                img1 = putText(img1, f"{result1['class_name']}, Confidence: {result1['confidence_score']}%", 10, 30)
                img1 = putText(img1, f"Emotion: {result1['emotion']}", 10, 70)
                img1 = putText(img1, f"Age: {result1_age}", 10, 110)
                img1 = putText(img1, f"Gender: {result1_gender} {result1_gender_confidence}%", 10, 150)
    # 如果當前幀的如果當前幀的的結果不需要處理更新，則則直接用前一次的結果           
    else: 
        img0 = putText(img0, f"{previous_results['class_name']}, Confidence: {previous_results['confidence_score']}%", 10, 30)
        img0 = putText(img0, f"Emotion: {previous_results['emotion']}", 10, 70)
        img0 = putText(img0, f"Age: {previous_results['age']}", 10, 110)
        img0 = putText(img0, f"Gender: {previous_results['gender']} {previous_results['gender_confidence']}%", 10, 150)
        
        img1 = putText(img1, f"{previous_results['class_name1']}, Confidence: {previous_results['confidence_score1']}%", 10, 30)
        img1 = putText(img1, f"Emotion: {previous_results['emotion1']}", 10, 70)
        img1 = putText(img1, f"Age: {previous_results['age1']}", 10, 110)
        img1 = putText(img1, f"Gender: {previous_results['gender1']} {previous_results['gender_confidence1']}%", 10, 150)

    # 如果類別2連續檢測超過3秒則退出循環
    if class_2_detected and start_time_2 and (time.time() - start_time_2) > 3:
        print("Class 2 detected for more than 3 seconds, stopping emotion analysis.")
        break
    out0.write(frame)
    out1.write(frame1)
    # 顯示攝像頭圖像
    cv2.imshow('camera0', img0)
    cv2.imshow('camera1', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_status = 1
        break

    frame_count += 1

# 釋放攝像頭,影片和關閉所有窗口
cap.release()
cap1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()

# 各情緒權重
basicpoint = 60
remain = 100 - basicpoint
negativeweight = -1
neutralweight = 0
positiveweight = 1

# 鏡頭一情緒轉換成1,0,-1
emotions_mapped0 = []
negative0 = 0
neutral0 = 0
positive0 = 0
for e in emotions_over_time:
    if e in emotion_categories['positive']:
        emotions_mapped0.append(1)
        positive0 += 1
    elif e in emotion_categories['negative']:
        emotions_mapped0.append(-1)
        negative0 += 1
    elif e in emotion_categories['neutral']:
        emotions_mapped0.append(0)
        neutral0 += 1

# 算鏡頭一各情緒百分比
total_emotions0 = negative0 + neutral0 + positive0
if total_emotions0 > 0:
    negative0perc = round(negative0 / total_emotions0, 2)
    neutral0perc = round(neutral0 / total_emotions0, 2)
    positive0perc = round(positive0 / total_emotions0, 2)
else:
    negative0perc = neutral0perc = positive0perc = 0

# 算鏡頭一分析後分數
cam0scr = basicpoint + negative0perc * remain * negativeweight + neutral0perc * remain * neutralweight + positive0perc * remain * positiveweight

# 鏡頭二情緒轉換成1,0,-1
emotions_mapped1 = []
negative1 = 0
neutral1 = 0
positive1 = 0
for e in emotions_over_time1:
    if e in emotion_categories['positive']:
        emotions_mapped1.append(1)
        positive1 += 1
    elif e in emotion_categories['negative']:
        emotions_mapped1.append(-1)
        negative1 += 1
    elif e in emotion_categories['neutral']:
        emotions_mapped1.append(0)
        neutral1 += 1

# 算鏡頭二各情緒百分比
total_emotions1 = negative1 + neutral1 + positive1
if total_emotions1 > 0:
    negative1perc = round(negative1 / total_emotions1, 2)
    neutral1perc = round(neutral1 / total_emotions1, 2)
    positive1perc = round(positive1 / total_emotions1, 2)
else:
    negative1perc = neutral1perc = positive1perc = 0

# 算鏡頭二分析後分數
cam1scr = basicpoint + negative1perc * remain * negativeweight + neutral1perc * remain * neutralweight + positive1perc * remain * positiveweight

# 以下動作將情緒映射為數值，並繪製情緒波動折線圖

# 將年齡和性別信息添加到圖表標題中
if ages_over_time and genders_over_time:
    avg_age = round(np.mean(ages_over_time))
    gender_counts = Counter(gender for gender, _ in genders_over_time)
    most_common_gender = gender_counts.most_common(1)[0][0]
    gender_confidence_avg = np.mean([confidence for _, confidence in genders_over_time])
    title_text = f"Emotion Wave Over Time in Cam0 (Avg Age: {avg_age}, Gender: {most_common_gender} {gender_confidence_avg:.2f}%)"
else:
    title_text = "Emotion Wave Over Time"

if ages_over_time1 and genders_over_time1:
    avg_age1 = round(np.mean(ages_over_time1))
    gender_counts1 = Counter(gender1 for gender1, _ in genders_over_time1)
    most_common_gender1 = gender_counts1.most_common(1)[0][0]
    gender_confidence_avg1 = np.mean([confidence1 for _, confidence1 in genders_over_time1])
    title_text1 = f"Emotion Wave Over Time in Cam1 (Avg Age: {avg_age1}, Gender: {most_common_gender1} {gender_confidence_avg1:.2f}%)"
else:
    title_text1 = "Emotion Wave Over Time"
    
Emotion_grade0 = 60+40*(positive0perc-negative0perc)
Emotion_grade1 = 60+40*(positive1perc-negative1perc)
print(f"Here is the Emotion Grade {Emotion_grade0} of Customer")
print(f"Here is the Emotion Grade {Emotion_grade1} of Server")


# 畫鏡頭一結合鏡頭二的折線圖 ！！這個地方尚未測試！！
emotions_map = [1 if e in emotion_categories['positive'] else -1 if e in emotion_categories['negative'] else 0 for e in emotions_over_time]
emotions_map1 = [1 if e1 in emotion_categories['positive'] else -1 if e1 in emotion_categories['negative'] else 0 for e1 in emotions_over_time1]

plt.figure(figsize=(10, 5))
plt.plot(emotions_map, label='Customer_Emotion_Wave', color='#1f77b4')
plt.plot(emotions_map1, label='Server_Emotion_Wave', color='#ff7f0e')
plt.axhline(y=0, color='gray', linestyle='--')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.title(title_text)
plt.xlabel("Frame")
plt.ylabel("Emotion")
plt.legend()
# 儲存折線圖為 JPG 文件
plt.savefig("Customer_Emotion_Wave & Server_Emotion_Wave.jpg", format='jpg')
# 儲存到特定路徑
plt.savefig("/Users/linjunting/Desktop/專題python/Customer_Emotion_Wave & Server_Emotion_Wave.jpg")

# 畫鏡頭一折線圖
emotions_mapped = [1 if e in emotion_categories['positive'] else -1 if e in emotion_categories['negative'] else 0 for e in emotions_over_time]
plt.figure(figsize=(10, 5))
plt.plot(emotions_mapped, label='Customer_Emotion_Wave', color='#1f77b4')
plt.axhline(y=0, color='gray', linestyle='--')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.title(title_text)
plt.xlabel("Frame")
plt.ylabel("Emotion")
plt.legend()
# 儲存折線圖為 JPG 文件
plt.savefig("Customer_Emotion_Wave.jpg", format='jpg')
# 儲存到特定路徑
plt.savefig("/Users/linjunting/Desktop/專題python/Customer_Emotion_Wave.jpg")


# 畫鏡頭一長條圖
emotions = ['Negative', 'Neutral', 'Positive']
percentages0 = [negative0perc, neutral0perc, positive0perc]
plt.figure(figsize=(8, 4))
bars0 = plt.bar(emotions, percentages0, color=['red', 'gray', 'green'])
plt.title(f'Combined Sentiment Analysis_Facial of Customer')
plt.xlabel('Sentiment')
plt.ylabel('Proportion')
plt.ylim(0, 1)
for bar in bars0:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=10, color='black')
# 儲存折線圖為 JPG 文件
plt.savefig("Customer_Emotion_Bar1.jpg", format='jpg')
# 儲存到特定路徑
plt.savefig("/Users/linjunting/Desktop/專題python/Customer_Emotion_Bar1.jpg")

# 畫鏡頭二折線圖
emotions_mapped = [1 if e1 in emotion_categories['positive'] else -1 if e1 in emotion_categories['negative'] else 0 for e1 in emotions_over_time1]
plt.figure(figsize=(10, 5))
plt.plot(emotions_mapped, label='Server_Emotion_Wave', color='#ff7f0e')
plt.axhline(y=0, color='gray', linestyle='--')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.title(title_text1)
plt.xlabel("Frame")
plt.ylabel("Emotion")
plt.legend()
# 儲存折線圖為 JPG 文件
plt.savefig("Server_Emotion_Wave.jpg", format='jpg')
# 儲存到特定路徑
plt.savefig("/Users/linjunting/Desktop/專題python/Server_Emotion_Wave.jpg")


# 畫鏡頭二長條圖
percentages1 = [negative1perc, neutral1perc, positive1perc]
plt.figure(figsize=(8, 4))
bars1 = plt.bar(emotions, percentages1, color=['red', 'gray', 'green'])
plt.title(f'Combined Sentiment Analysis_Facial of Serverq')
plt.xlabel('Sentiment')
plt.ylabel('Proportion')
plt.ylim(0, 1)
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=10, color='black')
# 儲存折線圖為 JPG 文件
plt.savefig("Server_Emotion_Bar.jpg", format='jpg')
# 儲存到特定路徑
plt.savefig("/Users/linjunting/Desktop/專題python/Server_Emotion_Bar.jpg")


# 返回 exit_status
if exit_status == 1:
    sys.exit('q')
else:
    sys.exit(0)