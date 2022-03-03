import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
df_data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = ['SepalLengthCm', 'SepalWidthCm', 
                     'PetalLengthCm', 'PetalWidthCm', 'Species'])

X = df_data.drop(labels = ['Species'],axis = 1).values 
# 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.3, random_state=42, stratify=y)

# 建立Logistic模型
logisticModel = LogisticRegression(random_state = 0)
# 使用訓練資料訓練模型
logisticModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = logisticModel.predict(X_train)

print('訓練集: ',logisticModel.score(X_train,y_train))
print('測試集: ',logisticModel.score(X_test,y_test))
