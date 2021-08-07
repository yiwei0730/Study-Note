# XGBoost 
###### tags: `study`

前情提要 : 是一個強大的機器學習方法，首先提到的是提升樹和GBDT的原理之後，在這裡使用（eXtreme Gradient Boosting）來完全加強


## 預先知道的知識
Regression Tree and Ensemble (What are we Learning，得到學習目標)
圖片取自網路
![](https://i.imgur.com/dOjALHj.png)
![](https://i.imgur.com/lLVTBj6.png)
每棵樹都一個函數，第二張圖就是將兩棵樹做合成(addition boosting)
其 Model: 
$\hat y_i = \sum^K_{k=1} f_k(x_i), f_k \in \mathcal{F}$
其 Objected function:
$OBj = \sum^n_{i=1}l(y_i,\hat y_i)+ \sum^K_{k=1}\Omega (f_k)$
The first is training loss, the second is the complexity of tree loss

## Gradient Boosting
1. How to learn(training)
```
We can not use methods such as SGD, to find f (since they are
trees, instead of just numerical vectors)
Solution: Additive Training (Boosting)
Start from constant prediction, add a new function each time
```
![](https://i.imgur.com/OVPDnBJ.png)
這種包含樹的模型不適合直接用SGD等優化演算法直接對整體模型進行優化，因而採用加法學習方法，boosting的學習策略是每次學習當前的樹，找到當前最佳的樹模型加入到整體模型中，因此關鍵在於學習第t棵樹。

2. Additive Training:定義目標函式，優化，尋找最佳的ft。
![](https://i.imgur.com/AG5V3rB.png)

在這裡$y_i$和 $\hat y^{(t-1)}_i$都是已知項，故所需要訓練的只剩下$f_t(x_i)$，而下圖是告訴我們殘差項的計算怎麼得來的

![](https://i.imgur.com/YYXvmGE.png)

3. Refine the definition of tree 重新定義每棵樹

• We define tree by a vector of scores in leafs, and a leaf index, mapping function that maps an instance to a leaf.
$q(x)$表示樣本$x$在某個葉子節點上，$w_{q(x)}$是該節點的打分,即該樣本的模型預測值
![](https://i.imgur.com/gxtCm9j.png)
4. Define the Complexity of Tree 樹的複雜度項
Define complexity as (this is not the only possible definition)
可以自訂，也可以使用此定義模式。
![](https://i.imgur.com/r2hmQWa.png)

5. Revisit the Objectives 更新
![](https://i.imgur.com/J62pwIo.png)

$T$ 是葉子個數, $w_{q(x)}$ 是節點分數，將其做項式統一，化解成全都是由i=1~T


6. The Structure Score
這個score是用來評價樹結構的。根據目標函數得到（見論文公式(4)、(5)、(6)），用於切分點查找算法。
![](https://i.imgur.com/tUu6EMW.png)
![](https://i.imgur.com/zvloSvB.png)
![](https://i.imgur.com/BtPsAhM.png)
切分分數越大反過來因為是負數代表我們需要的OBJ越小。

切分點查找算法 : 希望能夠找到越大越好(分數)的切分點, Gain還加了一個葉子節點複雜度項。
![](https://i.imgur.com/mPQD5ga.png)
![](https://i.imgur.com/6m1E2fI.png)
![](https://i.imgur.com/2T08xTt.png)

接著最後全部的算法經過
![](https://i.imgur.com/IsVWwhs.png)


## 與GDBT相比 XGBoost的優點

1. 原始的GDBT是以CART(tree)作為分類器，XGBoost則也可以使用並且有線性的分類器，通過booster[default=gbtree/gblinear] tree-based vs. linear

2. GDBT只使用到一階導數，XGBoost則使用到二階泰勒函數，並且使用一、二階導數，也可以自定義目標函數只要能夠使用一、二階導數即可

3. XGBoost在目標函數內還有使用正則化，對於正則項裡包含了樹的葉子節點個數、每個葉子節點上輸出的score的$L_2$的平方和。
從Bias-variance trade-off角度來講，正則項降低了模型variance，使學習出來的模型更加簡單，防止過擬合，這也是xgboost優於傳統GBDT的一個特性
4. shrinkage and column subsampling :
（1）shrinkage縮減類似於學習速率，在每一步tree boosting之後增加了一個參數$\epsilon$（權重），通過這種方式來減小每棵樹的影響力，給後面的樹提供空間去優化模型。
（2）column subsampling列(特徵)抽樣，說是從隨機森林那邊學習來的，防止過擬合的效果比傳統的行抽樣還好（行抽樣功能也有），並且有利於後面提到的並行化處理算法。

5. split finding algorithms(劃分點查找算法):
三種方法可參考此網頁https://blog.csdn.net/yilulvxing/article/details/106180900
(1) exact greedy algorithm : 貪心最優獲取切分點，此方法會列出所有的特徵，並以此特徵當作劃分條件，透過下面的式子把每個特徵所對應的參數帶入，值越大代表損失下降越多，並找出最好的劃分點。
(2) approximate algorithm :  近似完整、提出並統計的候選點概念。對於某個特徵，算法首先根據特徵分佈的分位數找到切割點的集合$S_k =\{s_{k1},s_{k2},\cdots,s_{kl}\}$，然後將特徵k的值根據集合Sk劃分到bucket中，接著對每個bucket的樣本統計值G、H進行累加統計，最後在這些累計的統計量上尋找最佳分裂點。
(3) Weighted Quantile Sketch : 分布式加權直方圖算法，可並行的近似直方圖算法。樹節點在進行分裂時，我們需要計算每個特徵的每個分割點對應的增益，即用貪心法枚舉所有可能的分割點。當數據無法一次載入內存或者在分佈式情況下，貪心算法效率就會變得很低，所以xgboost還提出了一種可並行的近似直方圖算法，用於高效地生成候選的分割點。為了解決數據無法一次載入內存或者在分佈式情況下（exact greedy algorithm）效率低的問題
https://medium.com/chung-yi/xgboost%E4%BB%8B%E7%B4%B9-b31f7ec8295e
6. 在本體的論文中有詳細解釋了對稀疏值的處理，Algorithm 3: Sparsity-aware Split Finding
7. Built-in Cross-Validation : 可以在裡面的迴圈去跑cross-validation
8. continue on Existing Model : 可以在已經跑過的模型後再去使用
9. High Flexibility :  可以是用自製的optimization去使用以及evaluation criteria.
10. 並行化處理 : 並非是訓練並行，而是特徵並行，決策樹的學習最耗時的一個步驟就是對特徵的值進行排序（因為要確定最佳分割點），xgboost在訓練之前，預先對數據進行了排序，然後保存為block結構，後面的迭代中重複地使用這個結構，大大減小計算量。這個block結構也使得併行成為了可能，在進行節點的分裂時，需要計算每個特徵的增益，最終選增益最大的那個特徵去做分裂，那麼各個特徵的增益計算就可以開多線程進行。

## 實戰設計
https://cyeninesky3.medium.com/xgboost-a-scalable-tree-boosting-system-%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98%E8%88%87%E5%AF%A6%E4%BD%9C-2b3291e0d1fe
```python=
pip install xgboost

import xgboost as xgb

# 分開數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1729)
print(X_train.shape, X_test.shape)

# 模型參數設定
xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)
# XGBClassifier == binary:logistic
xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)],early_stopping_rounds=100)

計算分數以及預測
preds = xlf.predict(X_test)
```

















