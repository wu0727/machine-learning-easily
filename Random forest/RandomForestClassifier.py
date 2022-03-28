import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
df_data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = ['SepalLengthCm', 'SepalWidthCm', 
                     'PetalLengthCm', 'PetalWidthCm', 'Species'])

X = df_data.drop(labels = ['Species'],axis = 1).values 
# 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
# 建立 Random Forest Classifier 模型，參數調整: n_estimators 森林中樹木的數量， gini 吉尼係數
randomForestModel = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
# 使用訓練資料訓練模型
randomForestModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = randomForestModel.predict(X_train)

# 預測成功的比例
print('訓練集: ',randomForestModel.score(X_train,y_train))
print('測試集: ',randomForestModel.score(X_test,y_test))

# 建立測試集的 DataFrme
df_test=pd.DataFrame(X_test, columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
df_test['Species'] = y_test
pred = randomForestModel.predict(X_test)
df_test['Predict'] = pred

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

sns.lmplot(x = "PetalLengthCm", y = "PetalWidthCm", hue = 'Species', data = df_test, fit_reg = False, legend = False)
plt.legend(title = 'target', loc = 'upper left', labels = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.title('訓練集: ' + str(randomForestModel.score(X_train,y_train)))
plt.show()
sns.lmplot(x = "PetalLengthCm", y = "PetalWidthCm", hue = "Predict", data = df_test, fit_reg = False, legend = False)
plt.legend(title = 'target', loc = 'upper left', labels = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.title('測試集: '+ str(randomForestModel.score(X_test,y_test)))
plt.show()
