import time
from dataframe import DataFrame
from model import Model

def main():
    # 計算程式執行時間
    ti = time.time()
    
    # 初始化 DataFrame 物件
    data = DataFrame()
    
    # 載入 iris 資料集
    data.load_dataset('iris')
    
    # 取得目標欄位並從資料中移除
    y =  data.get_column('target')
    data.drop_column('target')
    
    # 使用決策樹模型
    model = Model(data_x=data.get_dataframe(), data_y=y, model_type='dt', training_percent=0.8)
    
    # 訓練模型
    model.train()
    
    # 取得所有分類評估指標
    model.report()
    
    # 進行交叉驗證
    model.cross_validation(5)
   
    # 輸出程式執行時間
    print(time.time() - ti)

if __name__ == '__main__':
    main()
