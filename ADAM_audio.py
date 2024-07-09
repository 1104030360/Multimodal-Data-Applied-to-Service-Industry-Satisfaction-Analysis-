from pyannote.audio import Pipeline
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import os

# 設置 Hugging Face 的 API 存取令牌
your_auth_token = "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"

# 初始化Speaker分割管道
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=your_auth_token)

# 初始化情緒分析模型和特徵提取器
model_name = "HowMannyMore/wav2vec2-lg-xlsr-ur-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# 音頻文件路徑
audio_file_path = "C:/Users/anita/OneDrive/文件/畢業專題/畢業專題_音頻分割/新錄音 9.wav"
audio_input, sr = librosa.load(audio_file_path, sr=16000)

# 進行Speaker分割
diarization_result = diarization_pipeline({"audio": audio_file_path})

# 統計每個Speaker的出現次數，選擇出現次數最多的兩個Speaker
speaker_counts = Counter([speaker for _, _, speaker in diarization_result.itertracks(yield_label=True)])
top_speakers = [speaker for speaker, _ in speaker_counts.most_common(2)]

# 情緒類別映射
emotion_map = {
    'Negative': ['Anger', 'Disgust', 'Fearful', 'Sadness'],
    'Neutral': ['Neutral', 'Boredom'],
    'Positive': ['Happiness', 'Surprise']
}

# 初始化每個Speaker情緒得分的累積器
speaker_emotion_scores = defaultdict(lambda: defaultdict(list))

# 對每個Speaker的音頻片段進行情緒分析
min_length = 16000  # 假設最小長度為1秒

# 遍歷由Speaker分割結果中獲得的每個音頻片段
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    # 如果當前Speaker不在最頻繁出現的兩個Speaker之列，則跳過不處理
    if speaker not in top_speakers:
        continue  # 只分析兩個主要Speaker

    # 計算片段的開始和結束樣本索引
    start_sample = int(sr * segment.start)  # 將開始時間轉換成樣本索引
    end_sample = int(sr * segment.end)      # 將結束時間轉換成樣本索引
    segment_length = end_sample - start_sample  # 計算片段的長度

    # 如果片段長度小於設定的最小長度，則進行填充
    if segment_length < min_length:
        padding = min_length - segment_length  # 計算需要填充的樣本數
        # 對片段進行填充，保持長度一致
        speaker_segment = np.pad(audio_input[start_sample:end_sample], (0, padding), 'constant', constant_values=(0, 0))
    else:
        speaker_segment = audio_input[start_sample:end_sample]  # 否則直接使用原片段

    # 使用特徵提取器將音頻片段轉換為模型輸入格式
    inputs = feature_extractor(speaker_segment, return_tensors="pt", padding=True, sampling_rate=16000)
    # 禁用梯度計算，進行模型推理
    with torch.no_grad():
        logits = model(**inputs).logits  # 獲得未經softmax的輸出
    # 將logits通過softmax函數轉換為概率分布
    probs = torch.softmax(logits, dim=1)

    # 遍歷定義的情緒類別映射，計算各情緒類別的得分
    for category, labels in emotion_map.items():
        for label in labels:
            label_idx = int(model.config.label2id.get(label, -1))  # 從模型配置中獲得標籤索引
            if label_idx != -1:  # 確保標籤存在
                score = probs[0, label_idx].item()  # 獲取該標籤的概率得分
                # 將得分加入對應Speaker和情緒類別的得分列表中
                speaker_emotion_scores[speaker][category].append(score)

# 指定保存分數的路徑
scores_path = "C:/Users/anita/OneDrive/文件/畢業專題/畢業專題_音頻分割/Scores"
os.makedirs(scores_path, exist_ok=True)  # 確保文件夾存在
scores_file_path = os.path.join(scores_path, "scores.txt")

# 打開文件準備寫入
with open(scores_file_path, 'w') as file:
    # 為每個Speaker計算情緒得分的平均值
    for speaker, category_scores in speaker_emotion_scores.items():
        # 計算歸一化得分
        normalized_scores = {
            category: np.mean(scores) / sum(np.mean(scores) for scores in category_scores.values())
            for category, scores in category_scores.items()
        }

        # 根據指定公式計算加權得分
        weighted_score = 60 + (normalized_scores['Negative'] * -1 + normalized_scores['Neutral'] * 0.5 + normalized_scores['Positive'] * 1) * 40
        result = f"Weighted score for Speaker {speaker}: {weighted_score:.2f}\n"
        file.write(result)  # 將結果寫入文件

# 指定保存圖表的路徑
output_folder = "C:/Users/anita/OneDrive/文件/畢業專題/畢業專題_音頻分割/Results"
os.makedirs(output_folder, exist_ok=True)  # 確保文件夾存在

# 為每個Speaker計算情緒得分的平均值並繪圖
plt.figure(figsize=(6, 5))
width = 0.3  # 柱狀圖的寬度
x = np.arange(len(emotion_map))  # 情緒類別標籤位置

# 定義顏色
colors = ['#1f77b4', '#ff7f0e']  # 藍色和橘色

for i, (speaker, category_scores) in enumerate(speaker_emotion_scores.items()):
    # 根據新順序獲取平均分
    averages = [np.mean(scores) / sum(np.mean(scores) for scores in category_scores.values()) for category in ['Negative', 'Neutral', 'Positive'] for scores in [category_scores[category]]]
    plt.bar(x + i * width, averages, width, label=f"Speaker {speaker}", color=colors[i % len(colors)])

plt.xlabel('Sentiment')
plt.ylabel('Proportion')
plt.title('Combined Sentiment Analysis_Audio')
plt.xticks(x + width/2, ['Negative', 'Neutral', 'Positive'])
plt.legend()
plt.ylim(0, 1)

# 保存圖表到指定文件夾
plt.savefig(os.path.join(output_folder, "emotion_distribution_combined.png"))

plt.show()