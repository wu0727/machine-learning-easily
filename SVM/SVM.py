import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import sklearn as svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import make_meshgrid as mm
import plot_contours as plot
iris = load_iris()
df_data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = ['SepalLengthCm', 'SepalWidthCm', 
                     'PetalLengthCm', 'PetalWidthCm', 'Species'])

X = df_data.drop(labels = ['Species'], axis = 1).values 

# 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
from sklearn.decomposition import PCA
pca = PCA(n_components=2, iterated_power=1)
train_reduced = pca.fit_transform(X_train)
from sklearn import svm
# 建立 linearSvc 模型
linearSvcModel = svm.LinearSVC(C = 1, max_iter = 1000)
# 使用訓練資料訓練模型
linearSvcModel.fit(train_reduced, y_train)
# 使用訓練資料預測分類
predicted = linearSvcModel.predict(train_reduced)
# 計算準確率
accuracy = linearSvcModel.score(train_reduced, y_train)
pcaX0, pcaX1 = train_reduced[:, 0], train_reduced[:, 1]
#xx, yy = make_meshgrid(X0, X1)
MMx, MMy = mm.makemeshgrid(pcaX0, pcaX1)
plot.plot_contours(plt, linearSvcModel, MMx, MMy,
                  cmap = plt.cm.coolwarm, alpha = 0.8)

plt.scatter(pcaX0, pcaX1, c = y_train, cmap = plt.cm.coolwarm, s = 20, edgecolors = 'k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('LinearSVC (linear kernel) \n Accuracy: %.4f' % accuracy)
plt.show()
