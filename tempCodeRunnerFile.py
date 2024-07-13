import multiprocessing
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import os
import shutil
import webbrowser
from threading import Timer
import pdfkit

# 指定 wkhtmltopdf 的路径
path_wkhtmltopdf = '/usr/local/bin/wkhtmltopdf'  # 替换为你的路径
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

app = Flask(__name__)

# 模拟存储数据
data_store = {
    "time1": "",
    "time2": "",
    "name": "",
    "organization": "",
    "total_score": 0.0,
    "audio_score": 0.0,
    "text_score": 0.0,
    "facial_score": 0.0,
    "ai_text1": "",
    "ai_text2": "",
    "ai_text3": "",
    "person_photo": "", 
    "facial_chart": "", 
    "audio_chart": "", 
    "text_chart": "", 
}

def update_image_paths(name):
    img_folder = 'static/img'
    files = os.listdir(img_folder)
    for file in files:
        if f"person_photo_{name}" in file:
            data_store["person_photo"] = os.path.join(img_folder, file)  # 使用相对路径
        elif f"facial_chart_{name}" in file:
            data_store["facial_chart"] = os.path.join(img_folder, file)  # 使用相对路径
        elif f"audio_chart_{name}" in file:
            data_store["audio_chart"] = os.path.join(img_folder, file)  # 使用相对路径
        elif f"text_chart_{name}" in file:
            data_store["text_chart"] = os.path.join(img_folder, file)  # 使用相对路径


def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("CSV Data Loaded:")
        print(df)  # 打印读取的 CSV 数据框
        # 假设 CSV 文件包含与 data_store 对应的列
        for key in data_store.keys():
            if key in df.columns:
                print(f"Updating {key} with value {df[key].iloc[0]}")  # 打印更新的键和值
                data_store[key] = df[key].iloc[0]  # 取 CSV 文件中的第一行数据
        # 生成图片和图表路径
        name = data_store.get("name", "")
        if name:
            update_image_paths(name)

        print("Image paths updated:")
        print(f"person_photo: {data_store['person_photo']}")
        print(f"facial_chart: {data_store['facial_chart']}")
        print(f"audio_chart: {data_store['audio_chart']}")
        print(f"text_chart: {data_store['text_chart']}")
        
        print("Data store updated:", data_store)  # 打印更新后的 data_store
    except Exception as e:
        print(f"Error loading CSV file: {e}")

@app.route('/')
def report():
    """根路由，渲染报告页面"""
    return render_template('report2.html', data=data_store)

@app.route('/update', methods=['POST'])
def update_data():
    """更新数据的路由"""
    new_data = request.json
    data_store.update(new_data)
    return jsonify({"status": "success"})

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传并更新数据"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file and file.filename.endswith('.csv'):
        # 清空目标文件夹
        csv_folder_path = 'static/csv'
        img_folder_path = 'static/img'
        
        if os.path.exists(csv_folder_path):
            shutil.rmtree(csv_folder_path)
        os.makedirs(csv_folder_path, exist_ok=True)
        
        if os.path.exists(img_folder_path):
            shutil.rmtree(img_folder_path)
        os.makedirs(img_folder_path, exist_ok=True)

        filepath = os.path.join(csv_folder_path, file.filename)
        print(f"Saving file to {filepath}")
        file.save(filepath)
        load_csv_data(filepath)
        
        return redirect(url_for('report'))
    
    return jsonify({"status": "error", "message": "Invalid file type"})

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """生成 PDF 并下载"""
    try:
        # 使用绝对路径
        abs_data_store = data_store.copy()
        for key in ['person_photo', 'facial_chart', 'audio_chart', 'text_chart']:
            if abs_data_store[key]:
                abs_data_store[key] = os.path.abspath(abs_data_store[key])

        rendered = render_template('report2.html', data=abs_data_store)
        pdf_folder = 'static/pdf'
        pdf_filename = 'report2.pdf'
        pdf_path = os.path.join(pdf_folder, pdf_filename)

        # 确保目录存在
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)

        options = {
            'enable-local-file-access': None,  # 允许本地文件访问
            'page-size': 'A4',  # 设置页面大小
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
        }


        pdfkit.from_string(rendered, pdf_path, configuration=config, options=options)
        print(f"PDF saved to {pdf_path}")

        return send_file(pdf_path, as_attachment=True)
    except IOError as e:
        print(f"Error generating PDF: {e}")
        return "PDF 生成错误", 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # 在应用启动时加载预定义的 CSV 文件
    predefined_csv_path = os.path.join('static', 'csv', 'test-2.csv')
    if os.path.exists(predefined_csv_path):
        print(f"Loading predefined CSV file from {predefined_csv_path}")
        load_csv_data(predefined_csv_path)
    
    # 仅在主进程中启动浏览器
    if not os.getenv('WERKZEUG_RUN_MAIN'):
        Timer(1, open_browser).start()

    app.run(debug=True)
