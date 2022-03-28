import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
df_data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = ['SepalLengthCm', 'SepalWidthCm', 
                     'PetalLengthCm', 'PetalWidthCm', 'Species'])

X = df_data.drop(labels = ['Species'],axis = 1).values 
# 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
DTModel = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6, random_state = 42)
DTModel.fit(X_train, y_train)
predicted = DTModel.predict(X_train)
accuracy = DTModel.score(X_train, y_train)
print(accuracy)
