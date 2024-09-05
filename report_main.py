import multiprocessing
import subprocess
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import os
import shutil
import webbrowser
import json
from threading import Timer




app = Flask(__name__)


# Json檔抓到的數據
data_store = {
    "Manager_name": "",
    "Manager_organization": "",
    "Departmental_Information": "",
    "Organization_Name": "",
    "Service_Number": 0.0,
    "time1": "",
    "time2": "",
    "name": "",
    "organization": [],
    "total_score": 0.0,
    "audio_score": 0.0,
    "text_score": 0.0,
    "facial_score": 0.0,
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


data_store1 = {
    "Manager_name": "",
    "Manager_organization": "",
    "Departmental_Information": "",
    "Organization_Name": "",
    "Service_Number": 0.0,
    "time1": "",
    "time2": "",
    "name": "",
    "organization": [],
    "total_score": 0.0,
    "audio_score": 0.0,
    "text_score": 0.0,
    "facial_score": 0.0,
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





# 導入照片的地方
def update_image_paths(name):
    img_folder = 'static/img'
    files = os.listdir(img_folder)
    for file in files:
        if f"person_photo_{name}" in file:
            data_store["person_photo"] = os.path.join(img_folder, file)  
            
            
# 导入 staff1.json 数据
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
            update_image_paths(data_store1, name)

        print("Image paths updated:", data_store1)
        print("Data store updated:", data_store1)
    except Exception as e:
        print(f"Error loading Json file: {e}")
        
        
        
# 導入staff.json檔的地方        
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
                data_store["Average_audio_score"] = round(df["audio_score"].mean(), 1)
                data_store["Average_facial_score"] = round(df["facial_score"].mean(), 1)
                data_store["Average_text_score"] = round(df["text_score"].mean(), 1)
                data_store["Average_total_score"] = round(df["total_score"].mean(), 1)

        # 生成图片和图表路径
        name = data_store.get("name", "")
        if name:
            update_image_paths(name)

        print("Image paths updated:")
        print(f"person_photo: {data_store['person_photo']}")
        
        print("Data store updated:", data_store)  # 打印更新后的 data_store
    except Exception as e:
        print(f"Error loading Json file: {e}")


@app.route('/')
def report():
    """根路由，渲染报告页面"""
    print(f"Manager_name in report route: {data_store['Service_Number']}")
    return render_template('report.html', data=data_store, data1=data_store1)


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
        # 调用 Puppeteer 生成 PDF
        subprocess.run(["node", "generate_pdf.js"], check=True)
        
        return send_file("static/pdf/report1.pdf", as_attachment=True)
    except IOError as e:
        print(f"Error generating PDF: {e}")
        return "PDF 生成错误", 500
    
    
    
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")
    
    
if __name__ == '__main__':
    # 在应用启动时加载预定义的 CSV 文件
    predefined_json_path = os.path.join('static', 'json', 'staff.json')
    if os.path.exists(predefined_json_path):
        print(f"Loading predefined JSON file from {predefined_json_path}")
        load_json_data(predefined_json_path)
        
        predefined_json_path_staff = os.path.join('static', 'json', 'staff1.json')
    if os.path.exists(predefined_json_path_staff):
        print(f"Loading predefined JSON file from {predefined_json_path_staff}")
        load_json_data_staff(predefined_json_path_staff)
    
    # 仅在主进程中启动浏览器
    if not os.getenv('WERKZEUG_RUN_MAIN'):
        Timer(1, open_browser).start()

    app.run(debug=True)
