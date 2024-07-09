# ADAM_audio
## 安裝必要的軟體和 Python 庫
1. 安裝 Python：
   - 確保已安裝 Python 3.6 或更高版本。
2. 安裝 Python 庫：
   - 使用 pip 安裝所需的 Python 庫。打開命令行工具（如 cmd 或終端），並運行以下命令安裝必需的庫：
   - ```pip install matplotlib numpy librosa torch transformers pyannote.audio```
## 設定 Hugging Face API 令牌
3. 獲取 Hugging Face API 令牌：
   - 訪問 [Hugging Face 網站](https://huggingface.co/)，註冊或登錄後，創建一個新的 API 令牌。
   - 將獲得的令牌添加到您的程式碼中。在程式碼中找到 your_auth_token 變量並替換為您的實際令牌：
   - ```your_auth_token = "您的令牌"```
## 準備音頻檔案
4. 音頻檔案準備：
   - 確保有音頻檔案用於分析，並將音頻檔案的路徑更新到程式碼中相應的位置：
   - ```audio_file_path = "音頻文件路徑"```
## 執行程式
5. 運行程式：
   - 執行 ADAM_audio.py
## 查看結果
6. 查看結果：
   - 程式運行完畢後，檢查指定的輸出文件夾，會有生成的情緒分析圖表和分數文檔。
   - 圖表將保存為 emotion_distribution_combined.png，而分數將記錄在 scores.txt 文件中。
## 故障排除
7. 故障排除：
   - 如果遇到任何錯誤，請檢查 Python 庫是否安裝正確，API 令牌是否正確無誤，以及音檔路徑是否正確。
