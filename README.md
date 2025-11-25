ebm_interface/
├── UI/ # 前端介面檔案
├── ebm_app/ # 後端應用程式檔案
├── static/ # 靜態資源：圖像、CSS、JS
├── EBM_28.joblib # 訓練好的 EBM 模型檔案
├── Patient5.csv # 範例資料集或輸入樣本
├── db.sqlite3 # 若有內建資料庫
├── manage.py # 如為 Django 專案
└── talk_to_ebm.ipynb # Notebook 示範如何與模型互動

(與fetch_api同層
建立interface/data/temp.csv
醫院開container ，改view.py讀檔程式， pip intall -r > requirements.txt下載相關套件後， python manage.py runserver
