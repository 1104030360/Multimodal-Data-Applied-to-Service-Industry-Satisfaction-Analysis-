import requests
import json
import pandas as pd

# Power BI 配置信息
client_id = '10265bc8-a588-4034-b554-886134af0d79'
client_secret = 'ca7fb4b3-c0a9-4189-99e5-0553cbb5391e'
tenant_id = 'ab3ca549-6720-4beb-87e2-ee68221a6605'
group_id = 'YOUR_GROUP_ID'  # Power BI 工作区 ID
dataset_id = 'YOUR_DATASET_ID'  # Power BI 数据集 ID

# 获取访问令牌
def get_access_token():
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://analysis.windows.net/powerbi/api/.default'
    }
    response = requests.post(url, headers=headers, data=data)
    response_json = response.json()
    return response_json['access_token']

# 上传数据到 Power BI
def upload_data_to_powerbi(access_token, group_id, dataset_id, data):
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{group_id}/datasets/{dataset_id}/tables/TableName/rows"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print("Data uploaded successfully!")
    else:
        print(f"Failed to upload data. Response: {response.content}")

# 从后台获取数据示例
def get_data_from_backend():
    response = requests.get('BACKEND_API_URL')
    return response.json()

# 主函数
def main():
    access_token = get_access_token()
    data = get_data_from_backend()
    upload_data_to_powerbi(access_token, group_id, dataset_id, data)

if __name__ == "__main__":
    main()









def refresh_dataset(access_token, group_id, dataset_id):
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{group_id}/datasets/{dataset_id}/refreshes"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.post(url, headers=headers)
    if response.status_code == 202:
        print("Dataset refresh triggered successfully!")
    else:
        print(f"Failed to trigger dataset refresh. Response: {response.content}")

# 主函数中添加刷新数据集的步骤
def main():
    access_token = get_access_token()
    data = get_data_from_backend()
    upload_data_to_powerbi(access_token, group_id, dataset_id, table_name, data)
    refresh_dataset(access_token, group_id, dataset_id)

if __name__ == "__main__":
    main()
