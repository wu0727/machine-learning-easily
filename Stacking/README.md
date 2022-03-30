## 堆疊法 Stacking

堆疊法式結合許多獨立的模型所預測出來的結果<br>
將多個模型的輸出視為堆疊法模型的輸入<br>

由於每個模型預測出來的結果都不一樣，假設第二個模型預測結果較好，則能補足第一個模型的缺陷。<br>

分類:採投票制  回歸:加權平均法

---

Stacking 可以結合許多 model，在此範例中建立了四種迴歸器，
分別有隨機森林、支持向量機、KNN 與決策樹，
最終的模型採用兩層隱藏層的神經網路作為最後的房價預測評估模型；
測試的時候發現一層隱藏層以及三、四層隱藏層會增加 overfiting 的問題。

Parameters:

estimators: m 個model。
final_estimator: 集合所有model的輸出，訓練一個最終預測模型，
預設為LogisticRegression，範例中使用**MLP多層感知器**

Attributes:

estimators_: 查看 model 組合。
final_estimator: 查看最終整合訓練模型。
