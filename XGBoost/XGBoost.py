import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance

iris = load_iris()
df_data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = ['SepalLengthCm', 'SepalWidthCm', 
                     'PetalLengthCm', 'PetalWidthCm', 'Species'])
X = df_data.drop(labels = ['Species'],axis = 1).values 
# 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
# 建立 XGBClassifier 模型
xgboostModel = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
# 使用訓練資料訓練模型
xgboostModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = xgboostModel.predict(X_train)
print('訓練集: ', xgboostModel.score(X_train,y_train))
print('測試集: ', xgboostModel.score(X_test,y_test))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plot_importance(xgboostModel)
plt.title('特徵重要程度')
plt.show()
print('特徵重要程度: ', xgboostModel.feature_importances_)
# 建立測試集的 DataFrme
df_test=pd.DataFrame(X_test, columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
df_test['Species'] = y_test
pred = xgboostModel.predict(X_test)
df_test['Predict'] = pred
#訓練資料預測之圖形
sns.lmplot(x = "PetalLengthCm", y = "PetalWidthCm", hue = 'Species', data = df_test, fit_reg = False, legend = False)
plt.legend(title = 'target', loc = 'upper left', labels = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.title('訓練集: %d' % xgboostModel.score(X_train,y_train))
plt.show()
#測試資料預測之圖形
sns.lmplot(x = "PetalLengthCm", y = "PetalWidthCm", hue = "Predict", data = df_test, fit_reg = False, legend = False)
plt.legend(title = 'target', loc = 'upper left', labels = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.title('測試集: %.4f' % xgboostModel.score(X_test,y_test))
plt.show()

