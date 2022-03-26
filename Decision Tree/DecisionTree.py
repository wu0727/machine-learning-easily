# imports
import numpy as np
import matplotlib.pyplot as plt

# 亂數產生資料
np.random.seed(0)
noise = np.random.rand(100, 1)
x = np.random.rand(100, 1)
y = 3 * x + 15 + noise
# y=ax+b Target function  a=3, b=15


# plot
plt.scatter(x, y, s = 10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from sklearn.tree import DecisionTreeRegressor

# 建立DecisionTreeRegressor模型
decisionTreeModel = DecisionTreeRegressor(criterion = 'mse', max_depth = 4, splitter = 'best', random_state = 42)
# 使用訓練資料訓練模型
decisionTreeModel.fit(x, y)
# 使用訓練資料預測
predicted=decisionTreeModel.predict(x)

from sklearn import metrics
print('R2 score: ', decisionTreeModel.score(x, y))
mse = metrics.mean_squared_error(y, predicted)
print('MSE score: ', mse)
# plot
print('max_depth = 4,  MSE: ', mse)
plt.scatter(x, y, s = 10, label = 'True')
plt.scatter(x, predicted, color = "r",s = 10, label = 'Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
'''import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(decisionTreeModel, out_file = None, 
                         feature_names = ['x'],
                         filled = True, rounded = True,  
                         special_characters = True)  
graph = graphviz.Source(dot_data) 
print(graph)'''