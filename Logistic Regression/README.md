## 邏輯迴歸 (Logistic regression)<br>
---
邏輯回歸由線性回歸變化而來，是一種分類的模型ex.svm<br>
其目標是要找出一條線，能夠將資料做分類，因此可稱之為回歸的線性分類器<br>
利用`sigmoid function` 將輸出的值做0~1轉換<br>
當 output 大於0.5判定 1(假設)， output 小於0.5判定 0<br>

## 多元分類邏輯迴歸 (Multinomial Logistic Regression)<br>
---
多元分類有兩種方法，one-vs-rest(OvR) & many-vs-many(Mv1M)<br>
**OvR** :假設有ABC三種資料，A為正集，BC為負集，第二組為B正集，AC為負集，第三組為C正集，AB為負集，以此類推，看哪組預測分數比較高就決定該組別<br>

**MvM** :指挑兩個類別訓練一個分類器，因此有N個類別的資料就要N (N - 1) / 2個分類器，(A,B)(A,C)(B,C)，有新資料要預測時，以多數決的方法獲得預測結果。
