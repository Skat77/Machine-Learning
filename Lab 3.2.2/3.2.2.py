import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression

matplotlib.use('TkAgg')



iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='b', s=50, cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Многоклассовая логистическая регрессия')
plt.show()
