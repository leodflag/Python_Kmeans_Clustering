from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
"""
處理數據
"""
# 設定亂數
np.random.seed(42)
# 下載手寫資料集，分開資料X與標籤Y
X_digits_data, Y_digits_data = load_digits(return_X_y=True)
scale_data = scale(X_digits_data) # 資料z-score標準化，均值為0方差為1
n_samples, n_features = scale_data.shape # 1797, 64 = (1797,64)
# unique() 對一維數據去除重複元素，按元素小到大返回一個新的無元素重複的陣列
# 取得標籤數量
n_digits = len(np.unique(Y_digits_data))
labels = Y_digits_data # 原數據對應的標籤
sample_size = 300
"""
比較三種方法建立初始群聚中心(聚類標籤)的kmeans模型
k-means++、random、PCA-based
"""
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
# 衡量kmeans模型的效能
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data) # 將輸入建立好的模型放入資料訓練
    print('%-9s \t %.2fs \t %i \t %.3f \t %.3f \t %.3f \t%.3f \t%.3f \t%.3f'
                    %(name, (time() -t0), estimator.inertia_,
                    metrics.homogeneity_score(labels, estimator.labels_),
                    metrics.completeness_score(labels, estimator.labels_),
                    metrics.v_measure_score(labels, estimator.labels_),
                    metrics.adjusted_rand_score(labels, estimator.labels_),
                    metrics.adjusted_mutual_info_score(labels, estimator.labels_),
                    metrics.silhouette_score(data, estimator.labels_,
                                                                        metric = 'euclidean',
                                                                        sample_size=sample_size)))
# 建立好kmeans模型參數後丟入bench_k_means函數裡衡量效能
# 以k-means++ 的方式建立初始聚類中心
bench_k_means(KMeans(init = 'k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=scale_data)
# 從初始質心的數據中隨機選擇n_clusters個觀測值（列）
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=scale_data)

# 訓練一個主成份分析的模型，保留組件數為n_digits
pca = PCA(n_components=n_digits).fit(scale_data)
# pca.components_為array形式，shape (n_components, n_features)，給出初始中心
# 傳入pca特徵空間中的主軸（數據中最大方差方向）
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based", data=scale_data)
print(82*'_')
"""
將使用PCA降維數據訓練的kmeans模型訓練結果繪圖
"""
# fit_transform() 使用數據訓練PCA模型，得出降維（數量減少）後的數據
reduced_data = PCA(n_components=2).fit_transform(scale_data)
# 建立一個kmeans模型
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# 利用PCA降維的數據計算 Kmeans模型
kmeans.fit(reduced_data)
# 網格的步長
h = .02

# 將x與y的最大值與最小值界定出來。 [:, 0]取出矩陣col=0的直行資料
x_min, x_max = reduced_data[:, 0].min() -1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() -1, reduced_data[:, 1].max() + 1
# np.arange()會將兩數之間均勻間隔
# np.meshgrid()從座標向量返回座標矩陣，前者陣列擺放x軸複製len(y)個，後者將陣列擺放成y軸複製len(x)個
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 將xx , yy  使用ravel降為1維( row-style )，使用np.c_會將數據與標籤聯合起來，[[xx[0],yy[0]],[xx[1],yy[1]],...]
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# 將kmeans預測的Z重整成xx的長寬
Z = Z.reshape(xx.shape)

# 印出以PCA降維數據訓練kmeans模型預測出來的結果Z
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, #繪圖顏色為彩色
           aspect='auto', origin='lower')
#畫出數據點
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

centroids = kmeans.cluster_centers_
# 使用散點圖畫出kmeans的中心點
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max) # 設定x軸左右界
plt.ylim(y_min, y_max) # 設定y軸上下界
plt.xticks(())  # 獲取或設置x軸的當前刻度位置和標籤
plt.yticks(())  # 獲取或設置y軸的當前刻度位置和標籤
plt.show()





