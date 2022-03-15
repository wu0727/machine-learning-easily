import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
df_data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = ['SepalLengthCm', 'SepalWidthCm', 
                     'PetalLengthCm', 'PetalWidthCm', 'Species'])

X = df_data.drop(labels = ['Species'],axis = 1).values 
# 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 建立 KNN 模型
'''
n_neighbors: 設定k值, 也就是鄰居的數量, 預設為5。
algorithm: 搜尋數演算法{'auto', 'ball_tree', 'kd_tree', 'brute'}，可選。
metric: 計算距離的方式，預設為歐幾里得距離。
'''
KnnModel = KNeighborsClassifier(n_neighbors = 3)
# 使用訓練資料訓練模型
KnnModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = KnnModel.predict(X_train)



df_test = pd.DataFrame(X_test, columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
df_pred = df_test
df_test['Species'] = y_test
pred = KnnModel.predict(X_test)
df_test['Predict'] = pred

print('訓練集: ', KnnModel.score(X_train,y_train))
print('測試集: ', KnnModel.score(X_test,y_test))

sns.lmplot(x = "PetalLengthCm", y = "PetalWidthCm", data = df_test, hue = 'Species', fit_reg = False, legend = False)
plt.legend(title = 'Row data', loc = 'upper left', labels = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])

sns.lmplot(x = "PetalLengthCm", y = "PetalWidthCm", data = df_test, hue = 'Predict', fit_reg = False, legend = False)
plt.legend(title = 'predict data', loc = 'upper left', labels = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.show()
