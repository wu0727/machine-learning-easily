import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential

# 隨機定義新的x,y值
def make_data(N, err = 1, rseed = 21):
    rng = np.random.RandomState(rseed)
    x = rng.rand(N, 1) ** 2
    y = 10 - 1 / (x.ravel() + 0.1)
    if err > 0:
        y += err * rng.rand(N)
    return x, y

# 隨機產生100個 X 與 y
X, y = make_data(100)

# 建立 SGDRegressor 並設置超參數
regModel = SGDRegressor(max_iter = 100)
# 訓練模型
history = regModel.fit(X, y)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# 建立測試資料
x_test = np.linspace(-0.05,1,500)[:,None]
# 預測測試集

y_test=regModel.predict(x_test)
# 預測訓練集
y_pred=regModel.predict(X)
# 視覺化預測結果
plt.scatter(X,y)
plt.plot(x_test.ravel(), y_test, color="#d62728")
plt.xlabel('x')
plt.ylabel('y')
plt.text(0, 10, 'Loss(MSE) = %.3f' % mean_squared_error(y_pred, y), fontdict = {'size': 15, 'color':  'red'})
plt.show()

