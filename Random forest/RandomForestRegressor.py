# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn import metrics
# 亂數產生資料
np.random.seed(0)
noise = np.random.rand(100, 1)
x = np.random.rand(100, 1)
y = (3 * x + 15 + noise).ravel()
# y=ax+b Target function  a=3, b=15

# 建立RandomForestRegressor模型
randomForestModel = RFR(n_estimators = 100, criterion = 'mse')
# 使用訓練資料訓練模型
randomForestModel.fit(x, y)
# 使用訓練資料預測
predicted = randomForestModel.predict(x)

mse = metrics.mean_squared_error(y, predicted)
# plot
plt.scatter(x, y, s = 10, label = 'True')
plt.scatter(x, predicted, color = "r",s = 10, label = 'Predicted')
plt.title('R2 score: ' + str(randomForestModel.score(x, y)) + '\n' + 'MSE score: ' + str(mse))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
