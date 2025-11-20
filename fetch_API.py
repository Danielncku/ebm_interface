import datetime
import requests
import json
import pandas as pd

def getNowDate():
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    day_month_year = '{}-{}-{}'.format(year, month, day)
    # print('day_month_year: ' + day_month_year)
    return day_month_year

def getNowDatee():
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    day_month_year = '{}-{}-{} {}:{}'.format(year, month, day, hour, minute)
    # print('day_month_year: ' + day_month_year)
    return day_month_year

def getAPIResponse(day_month_year):
    url = 'http://10.11.29.18/php/dialysislist.php'
    param = {'date': day_month_year}

    ### API_Testing
    # url = 'https://jsonplaceholder.typicode.com/posts'
    # param = {}

    response = requests.get(url, params=param)
    response.raise_for_status()  # raises exception when not a 2xx response
    def get_now_date():
        """取得當前日期"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    """處理與保存資料"""
    if response.status_code == 200:
        try:
            # 去掉不必要的 meta 標籤，解析 JSON
            data = response.text.strip('<meta charset="UTF-8" />')
            data_list = json.loads(data)['data_list']

            # 動態生成檔案名稱，格式為 yyyy-mm.txt
            now = datetime.datetime.now()
            file_name = f"{now.year}-{now.month:02d}.txt"

            # 寫入文件
            with open(file_name, 'a') as file:
                file.write(f"Date: {get_now_date()}\n")
                file.write(data + "\n")
            
            print("Data saved successfully.")
        
        except Exception as error:
            data_list = []
            print("Error:", error)
    else:
        data_list = []
    return data_list

def convertCSV(data):
    col_name = ['ID', '姓名',	'性別',	'出生年月日',	'年齡',	'透析次數(本院)',	'透析開始時間',	'透析結束時間',	'紀錄時間',	'透析機編號',	'床位',	'體溫',	'開始體溫',	'透析前體重(kg)',	'理想體重(kg)',	'目標脫水量(L)',	'輸液量(L)',	'食物重量(kg)',	'預估脫水量(L)',	'設定脫水量(L)',	'結束體重(kg)',	'實際脫水量(L)',	'Start_SBP',	'Start_DBP',	'End_SBP',	'End_DBP',	'透析模式',	'透析器',	'開始透析液流速',	'開始血液流速',	'透析液Ca：3.0',	'傳導度：13.9',	'血管通路',	'Heparin',	'ESA',	'透析器凝血情況',	'血壓(收縮)',	'血壓(舒張)',	'脈搏',	'呼吸',	'血流速(ml/min)',	'透析液流速(ml/min)',	'靜脈壓(mmHg)',	'透析液壓(mmHg)',	'膜上壓(mmHg)',	'脫水速率',	'累積量',	'透析液溫度(℃)', '肝素注射量(ml/hr)',	'沖水量(L)',	'確認血管通路']
    res = pd.DataFrame.from_records(data, columns=col_name)
    res = res.sort_values(by=['ID', '透析開始時間', '透析結束時間', '紀錄時間'])
    row_indexes = res[res['床位'].apply(lambda x: x[0].isdigit())]
    res = res.drop(row_indexes.index)
    ### drop for patient with 2 IDs  ###
    row_indexes2 = res[res['ID'].apply(lambda x: x.isdigit()==False)]
    res = res.drop(row_indexes2.index)
    ####################################
    res = res.sort_values(by=['床位'], ascending=True)
    res.to_csv('interface/data/temp.csv', index=False, encoding='utf-8_sig')

def reorder():
    res = pd.read_csv('../data/temp.csv')
    res = res.sort_values(by=['床位'], ascending=True)
    row_indexes = res[res['床位'].apply(lambda x: x[0].isdigit())]
    res = res.drop(row_indexes.index)
    res.to_csv('interface/data/temp.csv', index=False, encoding='utf-8_sig', errors='ignore')

def run():
    date = getNowDate()
    data = getAPIResponse(date)
    convertCSV(data)

def fetchData():
    date = getNowDate()
    data = getAPIResponse(date)
    convertCSV(data)