# Python_Kmeans_Clustering

## 目標
	使用K-means分群(K-means clustering)對手寫數據集進行分群，其中以分群品質指標比較三種初始化群集中心的方法對資料的適合度。
  [code](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py)

## K-means 分群
	將n個無標籤的資料點分成k個群集，方式為每個資料點離某個分群中心點均值最近則為該群，並且群內平方和要最小
[wiki](https://zh.wikipedia.org/wiki/K-%E5%B9%B3%E5%9D%87%E7%AE%97%E6%B3%95)

### K-means公式
![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/debd28209802c22a6e6a1d74d099f728e6bd17a4)

### K-means演算法
	1.決定群集個數k，並挑選k個點作為群集中心
	2.每個資料點去比較跟k個群心的直線距離，最近的則分為該群
	3.重新計算各個群裡的群心
	4.重複2.、3.，直到群心不變為止

## 三種K-means初始化群心方法
	k-means++：以一種聰明方法選擇k-means的群集中心，以加快收斂速度
	random：從初始群心的數據中隨機選擇n_clusters個觀測值(列)
	ndarray：傳入形狀為(n_clusters，n_features)的array，並給出初始群心
[k-means++論文](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

## 主成份分析(Principal components analysis，PCA)
	在多元統計分析中，是一種統計分析、簡化數據集的方法。PCA會保留數據集中對變異數貢獻最大的特徵，
    作法是於數據集在訊息量最多的座標軸角度上投影，如利用正交矩陣或奇異值分解，得到一系列線性不相關變數的值，
    稱為主成份(Principal Components)，其可看成一個線性方程式，包含的線性係數可指示投影方向，如此便可用少量的
    主成份讓數據維度降低

### PCA演算法
    1.座標軸中心移到數據中心
    2.旋轉座標軸，使得數據在C1軸變異數最大(在此方向投影最為分散)，C1為第一主成份
    3.C2第二主成份，會與C1的共變異數(相關係數)為0，且盡量在C2軸變異數盡量最大，
    4.照3.步驟找到第p個主成份。

## 分群品質指標
	衡量K-means分群模型方法

### 分群品質指標簡稱表
![image](https://github.com/leodflag/Python_Kmeans_Clustering/blob/main/img/cluster_quality_metrics_shorthand.png)
### 分群品質指標列表
![image](https://github.com/leodflag/Python_Kmeans_Clustering/blob/main/img/cluster_quality_metrics_table.png)


## 虛擬碼
	1.下載手寫資料集，分開資料X與標籤Y
	2.將資料z-score標準化，平均值為0平方差為1
	3.建立bench_k_means函數，含分群品質指標表的衡量方式，比較原始標籤與預測標籤的分群品質
	4.用kmeans++、random兩種初始化群集中心方法建立兩種k-means模型
	5.使用PCA模型對標準化資料進行模型訓練，取出資料特徵空間主軸作為初始化群集中心方法建立k-means模型
	6.將4.、5.丟入3.bench_k_means函數，繪出圖表資料
	7.觀察結果，PCA降維數據訓練的k-means模型分群效果最好
	8.用PCA降維資料，建立並訓練k-means模型
	9.將結果繪圖


## 結果
### 分群品質指標結果比較表
![image](https://github.com/leodflag/Python_Kmeans_Clustering/blob/main/img/kmeans_clustering_evaluation.png)
### 使用PCA降維手寫數據集的K-means分群結果圖
![image](https://github.com/leodflag/Python_Kmeans_Clustering/blob/main/img/kmeans.png)
