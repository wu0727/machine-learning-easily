from operator import is_not
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as GBM
from sklearn.metrics import accuracy_score
import plot_matrix
#google 提供之信用卡盜刷資料集
def train_model(unbalance):
  raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
  X = raw_df.drop(columns = ['Class'])
  y = raw_df['Class']
  print('X:', X.shape)
  print('Y:', y.shape)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
  model = GBM.LGBMClassifier(is_unbalance = unbalance)
  model.fit(X_train, y_train)
  score = model.predict(X_test)
  print('score:', accuracy_score(y_test, score))
  if unbalance:
    plot_matrix.plot_confusion_matrix(y_test, score, "Is_Unbalance = True")
  else:
    plot_matrix.plot_confusion_matrix(y_test, score, "Is_Unbalance = False")
