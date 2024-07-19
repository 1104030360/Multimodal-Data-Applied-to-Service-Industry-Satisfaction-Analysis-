import multiprocessing
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import os
import shutil
import webbrowser
from threading import Timer
from weasyprint import HTML, CSS

app = Flask(__name__)

# 模拟存储数据
data_store = {
    "Manager_name": "",
    "Manager_organization":"",
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
    "Bar_total_summarize_text": "",  # 假设有一个对应总分的建议
    "Radar_text": "",
    "Pie_text": "",
    "Average_facial_score": 0.0,
    "Average_audio_score": 0.0,
    "Average_text_score": 0.0,
    "Average_total_score": 0.0,
}


def update_image_paths(name):
    img_folder = 'static/img'
    files = os.listdir(img_folder)
    for file in files:
        if f"person_photo_{name}" in file:
            data_store["person_photo"] = os.path.join(img_folder, file)  # 使用相对路径
            

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
        print(f"Error loading CSV file: {e}")
        

@app.route('/')
def report():
    """根路由，渲染报告页面"""
    return render_template('report.html', data=data_store)


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


# Flask app
@app.route('/api/get_ai_suggestion', methods=['GET'])
def get_ai_suggestion():
    score_type = request.args.get('type')
    suggestion_map = {
        "summarize_text1": data_store['Bar_facial_summarize_text'],
        "summarize_text2": data_store['Bar_audio_summarize_text'],
        "summarize_text3": data_store['Bar_text_summarize_text'],
        "summarize_text4": data_store['Bar_total_summarize_text']  # 假设有一个对应总分的建议
    }
    return jsonify(suggestion=suggestion_map.get(score_type, "No suggestion found")) # 給report_chart.js






@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """生成 PDF 并下载"""
    try:
        rendered = render_template('report2.html', data=data_store)
        pdf_folder = 'static/pdf'
        pdf_filename = 'report2.pdf'
        pdf_path = os.path.join(pdf_folder, pdf_filename)

        # 确保目录存在
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)

        # 渲染HTML和CSS
        html = HTML(string=rendered, base_url=request.url_root)
        css = CSS(filename=os.path.join('static', 'css', 'report2.css'))

        # 生成PDF
        html.write_pdf(pdf_path, stylesheets=[css])
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
