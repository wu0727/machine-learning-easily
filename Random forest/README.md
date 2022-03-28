## Random forest 隨機森林
由於決策樹於深度過大時，容易造成overfiting
因此衍伸出隨機森林，
該方法由多個決策樹組成，深度大之結果相較於決策樹也不易於overfiting
使用 Bagging 加上隨機特徵採樣。

#### 隨機森林之建立流程

1. 從訓練資料當中抽 n 筆資料(可重複抽取)，抽取 data 的方法又稱為 Bootstrap
2. 每組 data 中選擇 K 個特徵做為決策因子
3. 重複 m 次，產生 m 個決策樹
4. 分類: 多數決決定分類結果；回歸: 平均值決定預測值。

為何被稱呼**隨機森林**
- 抽取資料隨機且能重複抽取
- 特徵值隨機抽取

優點:
1. 每棵樹會用的 data & fecture 都隨機抽取
2. 多數決，利用多個決策樹的投票機制改善決策樹
3. VS. 決策樹，較不容易 overfiting
4. 森林中每棵樹都是獨立
5. 訓練或預測時，每棵樹都能平行運作。

---
RandomForestClassifier 常用之參數調整:
n_estimators: 樹木的數量，預設=100。
max_features: 劃分時考慮的最大特徵數，預設auto。
criterion: 亂度的評估標準，gini/entropy。預設為gini。
max_depth: 樹的最大深度。
splitter: 特徵劃分點選擇標準，best/random。預設為best。
random_state: 亂數種子，確保每次訓練結果都一樣，splitter = random 才有用。
min_samples_split: 至少有多少資料才能再分
min_samples_leaf: 分完至少有多少資料才能分

Attributes:
feature_importances_: 查詢模型特徵的重要程度。

Methods:
fit: 放入 x、y 進行模型訓練。
predict: 丟測試資料至訓練完的模型進行預測。
score: 預測成功的比例。
predict_proba: 預測每個類別的機率值。
get_depth: 取得樹的深度。
