import subprocess
import time
import requests

def start_flask_server():  # 先把網頁開起來
    # 启动 Flask 主进程
    print("Starting Flask server...")
    subprocess.Popen(['python', 'report2_main.py'])
    time.sleep(5)  # 等待 Flask 服务器启动

def start_server_action():  # 網頁開起來後送出POST請求
    # 发出 POST 请求
    data = {"action": "start"}
    try:
        response = requests.post('http://127.0.0.1:5002/perform-action', json=data)
        if response.status_code == 200:
            print("Response from server:", response.json())
        else:
            print(f"Failed to reach the server, status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # 启动 Flask 服务器
    start_flask_server()
    # 发送请求
    start_server_action()