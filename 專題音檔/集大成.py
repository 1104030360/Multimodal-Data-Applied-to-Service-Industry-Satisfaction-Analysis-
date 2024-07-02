from collections import defaultdict,Counter
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import whisper
from pyannote.audio import Pipeline
import numpy as np


# 设置 Hugging Face 访问令牌
your_auth_token = "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"

# 初始化说话人识别管道
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=your_auth_token)

# 检查CUDA是否可用，并据此设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# 加载 Whisper 模型
model = whisper.load_model("base")
model = model.to(device)

# 指定音频文件路径
audio_file_path = "C:/Users/蔡秉成/OneDrive/桌面/text/專題音檔/final.wav"

# 指定输出目录
output_dir = "C:/Users/蔡秉成/OneDrive/桌面/text/分割後樣本"
os.makedirs(output_dir, exist_ok=True)

# 使用 Pydub 加载音频
audio = AudioSegment.from_file(audio_file_path)

# 应用预训练的管道到你的音频文件进行说话人识别
diarization = pipeline(audio_file_path)
# 统计每个说话者的说话次数
speaker_counts = Counter(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
speakers_to_remove = set()

# 如果说话者超过两人，移除说话次数最少的说话者
while len(speaker_counts) > 2:
    # 找出说话次数最少的说话者
    speaker_to_remove = speaker_counts.most_common()[-1][0]
    del speaker_counts[speaker_to_remove]

# 过滤掉已删除的说话者的记录
filtered_diarization = [(turn, track, speaker) for turn, track, speaker in diarization.itertracks(yield_label=True) if speaker in speaker_counts]

# 初始化翻译器和情绪分析器
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

def translate(text, dest_language='en'):
    if not text:
        return "No text provided"
    try:
        result = translator.translate(text, dest=dest_language)
        if result and hasattr(result, 'text'):
            return result.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation failed"

def analyze_sentiment_vader(text):
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict

# 用于保存每个说话人情绪分析结果的字典
sentiments_by_speaker = defaultdict(list)
# 创建用于记录逐字稿的文件
script_file_path = os.path.join(output_dir, "transcript.txt")
with open(script_file_path, "w", encoding="utf-8") as script_file:

# 对每个说话人的语音片段进行识别、翻译和情绪分析
    for turn, _, speaker in filtered_diarization:
        start_ms = turn.start * 1000  # 转换为毫秒
        end_ms = turn.end * 1000
        segment = audio[start_ms:end_ms]

        # 如果说话者应被删除，则标记为unknown
        if speaker in speakers_to_remove:
            speaker_label = "unknown"
        else:
            speaker_label = speaker

    # 文件名使用speaker代码和时间戳
        segment_file_name = f"{speaker}_{int(start_ms)}_{int(end_ms)}.wav"
        segment_file_path = os.path.join(output_dir, segment_file_name)
        segment.export(segment_file_path, format="wav")

    # 只处理不被删除的说话者
        if speaker not in speakers_to_remove:
    # 使用 Whisper 进行语音识别
            result = model.transcribe(segment_file_path)

    # 翻译转录结果
            translated_text = translate(result['text'], 'en')

    # 记录逐字稿，包括时间戳
            script_file.write(f"Speaker {speaker} ({turn.start:.2f}-{turn.end:.2f}): {result['text']}\n")
            script_file.write(f"Translated: {translated_text}\n\n")
    
    # 打印转录结果和翻译结果
            print(f"Speaker {speaker}: {result['text']} (Original)")
            print(f"Translated: {translated_text}")

    # 进行情绪分析
            sentiment_dict = analyze_sentiment_vader(translated_text)
            sentiments_by_speaker[speaker].append(sentiment_dict)

    # 打印情绪分析结果
            print(f"Sentiment Analysis: {sentiment_dict}\n")

# 绘制合并的情绪分析图表
chart_output_dir = "C:/Users/蔡秉成/OneDrive/桌面/text/圖表"
os.makedirs(chart_output_dir, exist_ok=True)

labels = ['Negative', 'Neutral', 'Positive']
speakers = list(sentiments_by_speaker.keys())
width = 0.3  # 条形图的宽度

# 为每个说话者准备条形图数据
data = defaultdict(list)
for speaker, sentiments in sentiments_by_speaker.items():
    avg_sentiments = defaultdict(float)
    for sentiment in sentiments:
        for key, value in sentiment.items():
            avg_sentiments[key] += value / len(sentiments)
    data['neg'].append(avg_sentiments['neg'])
    data['neu'].append(avg_sentiments['neu'])
    data['pos'].append(avg_sentiments['pos'])

x = range(len(labels))  # 标签的位置

fig, ax = plt.subplots(figsize=(6, 5))
for idx, speaker in enumerate(speakers):
    ax.bar([p + width*idx for p in x], [data[key][idx] for key in ['neg', 'neu', 'pos']], width, alpha=0.7, label=f'Speaker {speaker}')

ax.set_xlabel('Sentiment')
ax.set_ylabel('Proportion')
ax.set_title('Combined Sentiment Analysis_TEXT')
ax.set_xticks([p + width*(len(speakers)-1)/2 for p in x])
ax.set_xticklabels(labels)
ax.legend()

# 設定 Y 軸範圍和刻度
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.1, 0.2))

# 调整布局并保存图表
plt.tight_layout()
chart_file_path = os.path.join(chart_output_dir, "combined_sentiment_analysis-TEXT.png")
plt.savefig(chart_file_path)
plt.close()
print("圖表已保存到", chart_file_path)

# 创建分数目录
scores_output_dir = "C:/Users/蔡秉成/OneDrive/桌面/text/分數"
os.makedirs(scores_output_dir, exist_ok=True)
# 计算分数并列出
scores = {}
for idx, speaker in enumerate(speakers):
    neg = data['neg'][idx]
    neu = data['neu'][idx]
    pos = data['pos'][idx]
    score = (neg * -1 + neu * 0.3 + pos * 1) * 40 + 60
    scores[speaker] = score

print("各Speaker的分数:")
for speaker, score in scores.items():
    print(f"{speaker}: {score:.2f}")

# 将分数保存到TXT文件
scores_file_path = os.path.join(scores_output_dir, "scores.txt")
with open(scores_file_path, "w", encoding="utf-8") as scores_file:
    for speaker, score in scores.items():
        scores_file.write(f"{speaker}: {score:.2f}\n")

print("各Speaker的分数已保存到", scores_file_path)
