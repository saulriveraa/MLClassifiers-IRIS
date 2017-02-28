 #!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from classifiers import *


c = np.genfromtxt('iris.data', delimiter = ',', dtype = None)
d = np.array([c['f0'], c['f3']]).T

e = np.zeros([len(d), 1])
e[50:100] = 1
e[100:150] = 2
e = np.ravel(e)


features_train = d
labels_train = e


clf, name = classifySVM(features_train, labels_train)
#clf, name = classifyNB(features_train, labels_train)


x_min = features_train[:,0].min(); x_max = features_train[:,0].max()
y_min = features_train[:,1].min(); y_max = features_train[:,1].max()

h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pcolormesh(xx, yy, Z, cmap='prism')

plt.scatter(features_train[:,0], features_train[:,1])
plt.title('Classification: ' + name)

plt.show()