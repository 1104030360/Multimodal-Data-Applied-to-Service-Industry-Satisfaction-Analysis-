<p align="center">
 <h2 align="center">Multimodal-Data-Applied-to-Service-Industry-Satisfaction-Analysis</h2>
 <p align="center"><img style="margin-bottom:-10px; height: 30px; width:30px;  " src="https://readme-components.vercel.app/api?component=logo&logo=react&fill=linear-gradient%2862deg%2C%20%238EC5FC%200%25%2C%20%23E0C3FC%20100%25%29%3B%0A&text=false&animation=spin"/>
 多模態資料應用於服務業滿意度分析
<img style="margin-bottom:-10px; height: 30px; width:30px;  margin-left: 10px;" src="https://readme-components.vercel.app/api?component=logo&logo=react&fill=linear-gradient%2862deg%2C%20%238EC5FC%200%25%2C%20%23E0C3FC%20100%25%29%3B%0A&text=false&animation=spin"/></p>
</p>
<hr>

#  簡介
這是我在畢業專題負責的部分（圖像辨識）

這個程式是一個實時的情緒檢測和分類有無在座位上的系統，它可以從攝像頭捕捉到的影像中檢測人臉，然後使用深度學習模型來分類該人臉所表達的情緒。

辨識完情緒後，按下Q鍵可以生成情緒起伏圖表，時間單位：一幀

此外可以透過判斷有無在座位上來決定情緒辨識的開始或結束，避免浪費儲存空間

*想看我更多的專案可以去這👉https://github.com/1104030360*

# 目前進度：

利用opencv結合deepface做出實時情緒辨識

利用matplotlib和numpy做出情緒起伏折線圖

利用訓練好的keras model辨識人有沒有在座位上以作為開始辨識情緒服務和結束辨識情緒服務的依據

此外成功實做出判斷顧客在位置上三秒後開始觸發情緒分析，並在離開座位三秒後結束情緒分析並匯出圖表之更方便使用者操作附加功能

實現外接網路攝影機並同時對兩隻攝影機捕捉到的畫面進行情緒分析，並在結束情緒分析服務後對兩種結果匯出兩種結果圖

持續更新中...


## 安裝要求
Python 

Keras

OpenCV

NumPy

deepface

Pillow

Matplotlib

## 安裝命令

```sh
pip install keras opencv-python numpy deepface pillow matplotlib
```

### 內容
模型載入和準備：從預先訓練好的 Keras 模型文件中載入情緒分類模型。

人臉檢測和情緒分析：使用 OpenCV 捕捉攝像頭的影像，並通過 deepface 庫對人臉進行情緒分析。

結果顯示：將分類結果和檢測到的情緒實時顯示在攝像頭的影像上。

情緒波浪圖：通過收集每一幀的情緒數據，將情緒隨時間變化的波動以波浪圖的形式展示出來。


![Pyhton progressbar](https://readme-components.vercel.app/api?component=linearprogress&value=100&skill=Python&fill=linear-gradient%2862deg%2C%20%238EC5FC%200%25%2C%20%23E0C3FC%20100%25%29%3B%0A)


