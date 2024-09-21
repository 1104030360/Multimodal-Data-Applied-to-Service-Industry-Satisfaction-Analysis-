import multiprocessing
import subprocess
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import os
import base64
import shutil
import webbrowser
import json
from threading import Timer





app = Flask(__name__)

# 定義保存圖片的目錄
UPLOAD_FOLDER = 'static/radar_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Admin Json檔抓到的數據
data_store = {
    "Admin_ID": "",
    "Manager_name": "",
    "Manager_organization": "",
    "Departmental_Information": "",
    "Organization_Name": "",
    "Service_Number": 0.0,
    "time1": "",
    "time2": "",
    "name": "",
    "organization": [],
    "Service_average_total_score": 0.0,
    "Service_average_audio_score": 0.0,
    "Service_average_text_score": 0.0,
    "Service_average_facial_score": 0.0,
    "Server_average_audio_score": 0.0,
    "Server_average_text_score": 0.0,
    "Server_average_facial_score": 0.0,
    "Server_average_total_score": 0.0,
    "ai_text1": "",
    "ai_text2": "",
    "ai_text3": "",
    "person_photo": "", 
    "Bar_facial_summarize_text": "",
    "Bar_audio_summarize_text": "",
    "Bar_text_summarize_text": "",
    "Bar_total_summarize_text": "",  
    "Radar_text": "",
    "Pie_text": "",
    "Average_facial_score": 0.0,
    "Average_audio_score": 0.0,
    "Average_text_score": 0.0,
    "Average_total_score": 0.0,
}

# Server Json檔抓到的數據
data_store1 = {
    "Server_ID": "",
    "name": "",
    "Service_Number": 0.0,
    "time1": "",
    "time2": "",
    "organization": [],
    "total_score": 0.0,
    "audio_score": 0.0,
    "text_score": 0.0,
    "facial_score": 0.0,
    "Average_facial_score": 0.0,
    "Average_audio_score": 0.0,
    "Average_text_score": 0.0,
    "Average_total_score": 0.0,
}





# 導入照片的地方
def update_image_paths(name):
    img_folder = 'static/img'
    files = os.listdir(img_folder)
    for file in files:
        if f"person_photo_{name}" in file:
            data_store["person_photo"] = os.path.join(img_folder, file)  
   

@app.route('/save_radar_image', methods=['POST'])
def save_radar_image():
    try:
        # 從請求中獲取圖片數據
        data = request.json['imageData']

        # 圖片數據是 base64 編碼的，我們需要去掉前綴並解碼
        image_data = data.split(',')[1]
        image_data = base64.b64decode(image_data)

        # 將圖片保存到 radar_images 文件夾，命名為 radar.png
        file_path = os.path.join(UPLOAD_FOLDER, 'radar.png')
        with open(file_path, 'wb') as f:
            f.write(image_data)

        return jsonify({"message": "Image saved successfully", "file_path": file_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500 
    
# 判斷是否是整數，整數的話不保留小數，否則保留一位小數
def format_score(score):
    return int(score) if score.is_integer() else round(score, 1)
    
   
# 導入Admin.json檔的地方        
def load_json_data(filepath):
    try:
        df = pd.read_json(filepath, orient='records')
        print("JSON Data Loaded:")
        print(df)  # 打印读取的 CSV 数据框
        for key in data_store.keys():
            if key in df.columns:
                print(f"Updating {key} with value {df[key].iloc[0]}")  # 打印更新的键和值
                if key == "Service_Number":
                    data_store[key] = round(df[key].iloc[0])  # 四捨五入為整數
                else:
                    data_store[key] = df[key].iloc[0]  # 取 CSV 文件中的第一行数据
                
                data_store["organization"] = ', '.join(df["organization"].tolist())  # 将所有组织信息合并为一个字符串                
                data_store["Average_audio_score"] = format_score(df["Service_average_audio_score"].mean())
                data_store["Average_facial_score"] = format_score(df["Service_average_facial_score"].mean())
                data_store["Average_text_score"] = format_score(df["Service_average_text_score"].mean())
                data_store["Average_total_score"] = format_score(df["Service_average_total_score"].mean())

        # 生成图片和图表路径
        name = data_store.get("name", "")
        if name:
            update_image_paths(name)

        print("Image paths updated:")
        print(f"person_photo: {data_store['person_photo']}")
        
        print("Data store updated:", data_store)  # 打印更新后的 data_store
    except Exception as e:
        print(f"Error loading Json file: {e}")
          
            
# 导入 Server.json 数据
def load_json_data_staff(filepath):
    try:
        df = pd.read_json(filepath, orient='records')
        print("JSON Data Loaded:")
        print(df)
        temp_data = {}
        for key in data_store1.keys():
            if key in df.columns:
                print(f"Updating {key} with value {df[key].iloc[0]}")
                if key == "Service_Number":
                    temp_data[key] = round(df[key].iloc[0])
                else:
                    temp_data[key] = df[key].iloc[0]
        temp_data["organization"] = ', '.join(df["organization"].tolist())
        temp_data["Average_audio_score"] = round(df["audio_score"].mean(), 1)
        temp_data["Average_facial_score"] = round(df["facial_score"].mean(), 1)
        temp_data["Average_text_score"] = round(df["text_score"].mean(), 1)
        temp_data["Average_total_score"] = round(df["total_score"].mean(), 1)
        data_store1.update(temp_data)

        name = data_store1.get("name", "")
        if name:
            update_image_paths(name)

        print("Image paths updated:", data_store1)
        print("Data store updated:", data_store1)
    except Exception as e:
        print(f"Error loading Json file: {e}")
        
        
        


@app.route('/')
def report():
    """根路由，渲染报告页面"""
    is_puppeteer = request.args.get('is_puppeteer', 'false').lower() == 'true'

    print(f"Manager_name in report route: {data_store['Service_Number']}")
    return render_template('report.html', data=data_store, data1=data_store1, is_puppeteer=is_puppeteer)


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
    
    if file and file.filename.endswith('.json'):
        # 清空目标文件夹
        json_folder_path = 'static/json'
        img_folder_path = 'static/img'
        
        if os.path.exists(json_folder_path):
            shutil.rmtree(json_folder_path)
        os.makedirs(json_folder_path, exist_ok=True)
        
        if os.path.exists(img_folder_path):
            shutil.rmtree(img_folder_path)
        os.makedirs(img_folder_path, exist_ok=True)

        filepath = os.path.join(json_folder_path, file.filename)
        print(f"Saving file to {filepath}")
        file.save(filepath)
        load_json_data(filepath)
        
        return redirect(url_for('report'))
    
    return jsonify({"status": "error", "message": "Invalid file type"})


# Flask app
@app.route('/api/get_ai_suggestion', methods=['GET'])
def get_ai_suggestion():
    score_type = request.args.get('type')
    suggestion_map = {
        "summarize_text1": data_store['Bar_facial_summarize_text'],
        "summarize_text2": data_store['Bar_audio_summarize_text'],
        "summarize_text3": data_store['Bar_text_summarize_text'],
        "summarize_text4": data_store['Bar_total_summarize_text']  
    }
    return jsonify(suggestion=suggestion_map.get(score_type, "No suggestion found")) # 給report_chart.js

# 生成PDF的地方
@app.route('/download_pdf1', methods=['GET'])
def download_pdf1():
    """生成 PDF 并下载"""
    try:
        # 提取 Admin_ID 作為文件名的一部分
        admin_id = data_store.get("Admin_ID", "default_id")  # 如果 Admin_ID 不存在，使用 'default_id'
        pdf_filename = f"{admin_id}.pdf"  # 將 Admin_ID 用作文件名
        pdf_folder = "static/pdf"
        pdf_path = os.path.join(pdf_folder, pdf_filename)  # 完整的 PDF 路徑
        

        # 調用 Puppeteer，並將 Admin_ID 作為參數傳遞給腳本
        process = subprocess.Popen(["node", "generate_pdf.js", admin_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # 检查 Puppeteer 是否成功运行
        if process.returncode == 0:
            if os.path.exists(pdf_path):
                return send_file(pdf_path, as_attachment=True)  # 返回生成的 PDF 文件
            else:
                return jsonify({"status": "error", "message": "PDF file not found"}), 404
        else:
            print(stderr.decode())  # 输出错误日志
            return jsonify({"status": "error", "message": "PDF generation failed"}), 500
  
    except IOError as e:
        print(f"Error generating PDF: {e}")
        return "PDF 生成错误", 500

@app.route('/perform-action', methods=['POST'])
def perform_action():
    data = request.json
    action_type = data.get('action')

    if action_type == 'start':
        result = "Server received 'start' action and performed the task."
        print(result)  # 在伺服器端顯示動作
    else:
        result = f"Server received unknown action: {action_type}"

    return jsonify({"message": result})  
    
    
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")
    
    
def start_process():
    # 在应用启动时加载预定义的 CSV 文件
    predefined_json_path_Customer = os.path.join('static', 'json', 'A1104036110403036.json')
    if os.path.exists(predefined_json_path_Customer):
        print(f"Loading predefined JSON file from {predefined_json_path_Customer}")
        load_json_data(predefined_json_path_Customer)
        
        predefined_json_path_Server = os.path.join('static', 'json', 'Server.json')
    if os.path.exists(predefined_json_path_Server):
        print(f"Loading predefined JSON file from {predefined_json_path_Server}")
        load_json_data_staff(predefined_json_path_Server)
    
    # 仅在主进程中启动浏览器
    if not os.getenv('WERKZEUG_RUN_MAIN'):
        Timer(1, open_browser).start()

    app.run(debug=True)
