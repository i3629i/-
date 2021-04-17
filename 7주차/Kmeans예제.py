{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "\n",
    "from sklearn.cluster import KMeans\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  \\\n",
      "0      1            5.1           3.5            1.4           0.2   \n",
      "1      2            4.9           3.0            1.4           0.2   \n",
      "2      3            4.7           3.2            1.3           0.2   \n",
      "3      4            4.6           3.1            1.5           0.2   \n",
      "4      5            5.0           3.6            1.4           0.2   \n",
      "..   ...            ...           ...            ...           ...   \n",
      "145  146            6.7           3.0            5.2           2.3   \n",
      "146  147            6.3           2.5            5.0           1.9   \n",
      "147  148            6.5           3.0            5.2           2.0   \n",
      "148  149            6.2           3.4            5.4           2.3   \n",
      "149  150            5.9           3.0            5.1           1.8   \n",
      "\n",
      "            Species  \n",
      "0       Iris-setosa  \n",
      "1       Iris-setosa  \n",
      "2       Iris-setosa  \n",
      "3       Iris-setosa  \n",
      "4       Iris-setosa  \n",
      "..              ...  \n",
      "145  Iris-virginica  \n",
      "146  Iris-virginica  \n",
      "147  Iris-virginica  \n",
      "148  Iris-virginica  \n",
      "149  Iris-virginica  \n",
      "\n",
      "[150 rows x 6 columns]\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv('./kmeans_data/Iris.csv')\n",
    "print(data)\n",
    "X = data.iloc[:,[1,2]].values\n",
    "#2열,3열 데이터 자름\n",
    "# print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[130.18093333333334, 57.98240604207882, 37.1237021276596, 27.98254281735862, 20.949686646361823, 17.20830189701896, 14.628929179122284, 13.0366209882072, 11.16093971237741, 10.10973969508752, 9.022971089041684, 7.591740916462278, 6.935564353946706, 6.518378729468884, 6.048248177476118, 5.706324986417095, 5.249864073834665, 4.933097763347765, 4.353052579773168]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "wcss = []\n",
    "# n_cluster 클러스터의 중심수\n",
    "# random_state 시드값\n",
    "# max_iter최대 반복 수(기본값300)\n",
    "# n_init 초기 클러스터의 중심위치 시도횟수 (기본값10)\n",
    "\n",
    "#wcss를 통해 k개의 클러스터를 좀더 확실하게 정하 수 있다고 하는데..잘 모르겠다..\n",
    "# Elbow Method - 클러스터의 수를 순차적으로 늘려가면서 결과 모니터링\n",
    "# within group에서 그래프에서 기울기가 완만해 지는 곳을 ElbowPoint라해서 이K가 적정값이라 판단.\n",
    "\n",
    "for i in range(1, 20):\n",
    "    kmeans = KMeans(n_clusters= i, init = 'k-means++', random_state = 50)\n",
    "    kmeans.fit(X)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "    plt.scatter(i, kmeans.inertia_, s=50, c='r')\n",
    "print(wcss)\n",
    "plt.plot(range(1,20), wcss)\n",
    "plt.title('Elbow Method')\n",
    "plt.xlabel(\"Number of cluster\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[5.006      3.418     ]\n",
      " [5.77358491 2.69245283]\n",
      " [6.81276596 3.07446809]]\n",
      "[5.1 4.9 4.7 4.6 5.  5.4 4.6 5.  4.4 4.9 5.4 4.8 4.8 4.3 5.8 5.7 5.4 5.1\n",
      " 5.7 5.1 5.4 5.1 4.6 5.1 4.8 5.  5.  5.2 5.2 4.7 4.8 5.4 5.2 5.5 4.9 5.\n",
      " 5.5 4.9 4.4 5.1 5.  4.5 4.4 5.  5.1 4.8 5.1 4.6 5.3 5.  7.  6.4 6.9 5.5\n",
      " 6.5 5.7 6.3 4.9 6.6 5.2 5.  5.9 6.  6.1 5.6 6.7 5.6 5.8 6.2 5.6 5.9 6.1\n",
      " 6.3 6.1 6.4 6.6 6.8 6.7 6.  5.7 5.5 5.5 5.8 6.  5.4 6.  6.7 6.3 5.6 5.5\n",
      " 5.5 6.1 5.8 5.  5.6 5.7 5.7 6.2 5.1 5.7 6.3 5.8 7.1 6.3 6.5 7.6 4.9 7.3\n",
      " 6.7 7.2 6.5 6.4 6.8 5.7 5.8 6.4 6.5 7.7 7.7 6.  6.9 5.6 7.7 6.3 6.7 7.2\n",
      " 6.2 6.1 6.4 7.2 7.4 7.9 6.4 6.3 6.1 7.7 6.3 6.4 6.  6.9 6.7 6.9 5.8 6.8\n",
      " 6.7 6.7 6.3 6.5 6.2 5.9]\n",
      "[3.5 3.  3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 3.7 3.4 3.  3.  4.  4.4 3.9 3.5\n",
      " 3.8 3.8 3.4 3.7 3.6 3.3 3.4 3.  3.4 3.5 3.4 3.2 3.1 3.4 4.1 4.2 3.1 3.2\n",
      " 3.5 3.1 3.  3.4 3.5 2.3 3.2 3.5 3.8 3.  3.8 3.2 3.7 3.3 3.2 3.2 3.1 2.3\n",
      " 2.8 2.8 3.3 2.4 2.9 2.7 2.  3.  2.2 2.9 2.9 3.1 3.  2.7 2.2 2.5 3.2 2.8\n",
      " 2.5 2.8 2.9 3.  2.8 3.  2.9 2.6 2.4 2.4 2.7 2.7 3.  3.4 3.1 2.3 3.  2.5\n",
      " 2.6 3.  2.6 2.3 2.7 3.  2.9 2.9 2.5 2.8 3.3 2.7 3.  2.9 3.  3.  2.5 2.9\n",
      " 2.5 3.6 3.2 2.7 3.  2.5 2.8 3.2 3.  3.8 2.6 2.2 3.2 2.8 2.8 2.7 3.3 3.2\n",
      " 2.8 3.  2.8 3.  2.8 3.8 2.8 2.8 2.6 3.  3.4 3.1 3.  3.1 3.1 3.1 2.7 3.2\n",
      " 3.3 3.  2.5 3.  3.4 3. ]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "n_clusters=3\n",
    "Kmean = KMeans(n_clusters)\n",
    "Kmean.fit(X)\n",
    "center = Kmean.cluster_centers_\n",
    "print(center)\n",
    "print(X[:,0])\n",
    "print(X[:,1])\n",
    "plt.scatter(X[:,0], X[:,1], s=20)\n",
    "for i in range(n_clusters):\n",
    "    plt.scatter(center[i][0], center[i][1],s=50, c='r', marker='*')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "wonseok",
   "language": "python",
   "name": "wonseok"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
