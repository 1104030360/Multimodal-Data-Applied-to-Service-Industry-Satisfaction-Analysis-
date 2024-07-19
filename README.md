# 程式碼概述_ADAM text
此程式碼用於分析使用者對話文本的情緒，程式碼運行步驟如下:
1. 進行說話人辨識並過濾音訊檔。
2. 使用 Whisper 模型對音訊進行語音轉文字。
3. 翻譯轉寫的文字。
4. 透過VADAR進行情緒分析。
5. 繪製情緒分析結果的圖表。
6. 計算並保存每個說話者的情緒分析分數。

# 必要的安裝套件
安裝此程式前，需要確保安裝以下Python套件：

pip install numpy matplotlib torch pydub vaderSentiment googletrans==4.0.0-rc1 whisper pyannote.audio

# 特別注意
- Whisper 和 pyannote.audio 套件需要額外的模型下載，可能需要較大的網絡流量消耗。
- 由於 googletrans 是第三方套件，其穩定性可能會受到 Google 服務更改的影響。

# 輸出文件
- 分割後的語音檔：保存在資料夾'分割後樣本'中。
- 顧客與服務人員的對話劇本：保存成`transcript.txt`並存入資料夾'分割後樣本'中。
- 情緒分析圖表：保存成'combined_sentiment_analysis-TEXT.png'並存入資料夾'圖表'中。
- 情緒分數：保存成 `scores.txt`並存入資料夾'分數'中。
