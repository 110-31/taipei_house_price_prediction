import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 路徑設定
DATA_PATH = './data/Taipei_house.csv'
MODEL_PATH = './models/house_price_model.pkl'

# 1. 讀取資料
data = pd.read_csv(DATA_PATH)
print("資料快速查看:\n", data.head())

# 2. 資料預處理
# (處理缺失值)
data = data.dropna()

# 3. 特徵與目標值設定
# (假設 'price' 是房價，其餘為特徵)
X = data.drop('price', axis=1)
y = data['price']

# 4. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 建立模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 預測與模型評估
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n模型評估結果:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}")

# 7. 儲存模型
os.makedirs('./models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\n模型已成功儲存至 {MODEL_PATH}")

