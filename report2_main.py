import multiprocessing
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import requests
import pandas as pd
import os
import shutil
import webbrowser
from threading import Timer
import json
from weasyprint import HTML, CSS

app = Flask(__name__)

# Customer
data_store = {
    #  這邊接入平均分數
    "average_total_score": 0.0,
    "average_facial_score": 0.0,
    "average_audio_score": 0.0,
    "average_text_score": 0.0,

    "time1": "",
    "time2": "",
    "name": "",
    "Service_ID": "",
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
    "audio_color": "",
    "text_color": "",
    "facial_color": "",
}
# Server
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
    "Average_audio_color": "",
    "Average_text_color": "",
    "Average_facial_color": "",
}


def update_image_paths(data, Service_ID):
    img_folder = 'static/img'
    files = os.listdir(img_folder)
    for file in files:
        if f"{Service_ID}_person_photo" in file:
            data["person_photo"] = os.path.join(img_folder, file)
        elif f"{Service_ID}_facial_chart" in file:
            data["facial_chart"] = os.path.join(img_folder, file)
        elif f"{Service_ID}_audio_chart" in file:
            data["audio_chart"] = os.path.join(img_folder, file)
        elif f"{Service_ID}_text_chart" in file:
            data["text_chart"] = os.path.join(img_folder, file)
            
def generate_color(score):
    return '#4ef973' if score > 70 else '#fcbe7f'


# 导入 staff.json 数据
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
        temp_data['audio_color'] = generate_color(temp_data.get('audio_score', 0))
        temp_data['text_color'] = generate_color(temp_data.get('text_score', 0))
        temp_data['facial_color'] = generate_color(temp_data.get('facial_score', 0))


        data_store1.update(temp_data)

        ID_Server = data_store1.get("ID_Server", "")
        if ID_Server:
            update_image_paths(data_store1, ID_Server)
            
        print("Image paths updated:", data_store1)
        print("Data store updated:", data_store1)
    except Exception as e:
        print(f"Error loading Json file: {e}")

# 导入 server.json 数据
def load_json_data(filepath):
    try:
        df = pd.read_json(filepath, orient='records')
        print("JSON Data Loaded:")
        print(df)
        temp_data = {}
        for key in data_store.keys():
            if key in df.columns:
                print(f"Updating {key} with value {df[key].iloc[0]}")
                temp_data[key] = df[key].iloc[0]
                
        temp_data["Average_audio_color"] = generate_color(temp_data.get('Average_audio_score', 0))
        temp_data["Average_text_color"] = generate_color(temp_data.get('Average_text_score', 0))
        temp_data["Average_facial_color"] = generate_color(temp_data.get('Average_facial_score', 0))
        data_store.update(temp_data)
            
        Service_ID = data_store.get("Service_ID", "")
        if Service_ID:
            update_image_paths(data_store, Service_ID) 
                   
        print("Image paths updated:", data_store)
        print("Data store updated:", data_store)
    except Exception as e:
        print(f"Error loading JSON file: {e}")

@app.route('/')
def report():
    """根路由，渲染报告页面"""
    return render_template('report2.html', data=data_store, data1=data_store1)

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
    
    if file and file.filename.endswith('.json'):  # 修正 .json 文件后缀
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

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """生成 PDF 并下载"""
    try:
        rendered = render_template('report2.html', data=data_store, data1=data_store1)
        pdf_folder = 'static/pdf'
        pdf_filename = f'{data_store.get("Service_ID", "default_id")}.pdf'
        pdf_path = os.path.join(pdf_folder, pdf_filename)

        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)

        html = HTML(string=rendered, base_url=request.url_root)
        css = CSS(filename=os.path.join('static', 'css', 'report2.css'))

        html.write_pdf(pdf_path, stylesheets=[css])
        print(f"PDF saved to {pdf_path}")

        return send_file(pdf_path, as_attachment=True)
    except IOError as e:
        print(f"Error generating PDF: {e}")
        return "PDF 生成错误", 500


# 修改后的 perform-action 路由
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
    
    predefined_json_path_Customer = os.path.join('static', 'json', 'S1104036110403036.json')
    if os.path.exists(predefined_json_path_Customer):
        print(f"Loading predefined JSON file from {predefined_json_path_Customer}")
        load_json_data(predefined_json_path_Customer)
        
    predefined_json_path_Server = os.path.join('static', 'json', 'Server.json')
    if os.path.exists(predefined_json_path_Server):
        print(f"Loading predefined JSON file from {predefined_json_path_Server}")
        load_json_data_staff(predefined_json_path_Server)

    if not os.getenv('WERKZEUG_RUN_MAIN'):
        Timer(1, open_browser).start()

    app.run(debug=True)