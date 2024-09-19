import requests
import subprocess
import time

def start_flask_server():
    # 使用 subprocess 啟動 Flask 主程式，並保持它在後台運行
    print("Starting Flask server...")
    process = subprocess.Popen(['python', 'report2_main.py'])  
    time.sleep(5)  # 等待伺服器啟動

def start_server_action():
    # 要發送的資料
    data = {"action": "start"}
    
    # 向主程式發送 POST 請求
    try:
        response = requests.post('http://127.0.0.1:5000/perform-action', json=data)
        
        # 檢查請求結果
        if response.status_code == 200:
            print("Response from server:", response.json())
        else:
            print(f"Failed to reach the server, status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # 啟動 Flask 伺服器
    start_flask_server()

    # 等伺服器啟動後，發送請求
    start_server_action()
