from pyannote.audio import Pipeline
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from collections import Counter

# 设置 Hugging Face 访问令牌
your_auth_token = "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"

# 初始化说话人分割管道
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=your_auth_token)

# 初始化情绪分析模型和特征提取器
model_name = "HowMannyMore/wav2vec2-lg-xlsr-ur-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# 音频文件路径
audio_file_path = "C:/Users/anita/OneDrive/文件/畢業專題/畢業專題_音頻分割/新錄音 9.wav"
audio_input, sr = librosa.load(audio_file_path, sr=16000)

# 进行说话人分割
diarization_result = diarization_pipeline({"audio": audio_file_path})

# 统计每个说话人的出现次数并选择最频繁的两个
speaker_counts = Counter([speaker for _, _, speaker in diarization_result.itertracks(yield_label=True)])
top_speakers = [speaker for speaker, _ in speaker_counts.most_common(2)]

# 情绪类别映射
emotion_map = {
    'Positive': ['Happiness', 'Surprise'],
    'Neutral': ['Neutral', 'Boredom'],
    'Negative': ['Anger', 'Disgust', 'Fearful', 'Sadness']
}

# 初始化每个说话人情绪得分的累积器
speaker_emotion_scores = defaultdict(lambda: defaultdict(list))

# 对每个说话人的音频片段进行情绪分析
min_length = 16000  # 假设最小长度为1秒

for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    if speaker not in top_speakers:
        continue  # 只分析两个主要说话人
    start_sample = int(sr * segment.start)
    end_sample = int(sr * segment.end)
    segment_length = end_sample - start_sample

    if segment_length < min_length:
        padding = min_length - segment_length
        speaker_segment = np.pad(audio_input[start_sample:end_sample], (0, padding), 'constant', constant_values=(0, 0))
    else:
        speaker_segment = audio_input[start_sample:end_sample]

    inputs = feature_extractor(speaker_segment, return_tensors="pt", padding=True, sampling_rate=16000)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)

    for category, labels in emotion_map.items():
        for label in labels:
            label_idx = int(model.config.label2id.get(label, -1))
            if label_idx != -1:
                score = probs[0, label_idx].item()
                speaker_emotion_scores[speaker][category].append(score)

# 为每个说话人计算情绪得分的平均值并绘图
for speaker, category_scores in speaker_emotion_scores.items():
    labels, averages = [], []
    total_sum = sum(np.mean(scores) for scores in category_scores.values())  # 计算归一化的分母

    print(f"Results for Speaker {speaker}:")
    for category, scores in category_scores.items():
        average_score = np.mean(scores) / total_sum  # 归一化每个情绪类别的得分
        labels.append(category)
        averages.append(average_score)
        print(f"  {category}: {average_score:.2f}")

    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, averages, color=['green', 'gray', 'red'])
    plt.title(f"Speaker {speaker} - Normalized Emotion Distribution")
    plt.ylabel("Normalized Average Probability")
    plt.ylim(0, 1)
    plt.show()
