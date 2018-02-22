# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:21:04 2018

@author: J
"""

from __future__ import print_function
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False 
#在控制台显示日志
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
lfw_people=fetch_lfw_people(min_faces_per_person=70,resize=0.4)
n_samples,h,w=lfw_people.images.shape
X=lfw_people.data
n_features=X.shape[1]       #每个像素代表一个特征值
Y=lfw_people.target
target_names=lfw_people.target_names
n_classes=target_names.shape[0]
print("*************总数据集大小**********************")
print("**图数量: %d" % n_samples)
print("**特征向量数: %d" % n_features)
print("**人数: %d" % n_classes)
print("*************总数据集大小**********************")
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
#特征提取/降维
n_components=150
print("从 %d 个维度中提取到 %d 维度" % (X_train.shape[0],n_components))
#主成分建模
pca=PCA(n_components=n_components,whiten=True).fit(X_train)
eignfaces=pca.components_.reshape((n_components,h,w))
print("根据主成分进行降维开始")
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)
print("降维结束")
#训练SVM分类
print("训练SVM分类模型开始")
t0=time()
#构建归类精确度5*6=30
param_grid={'C':[1e3,5e3,1e4,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1],} #实验字典
clf=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
clf=clf.fit(X_train_pca,Y_train)
print("SVM训练结束，结果如下：" "SVM训练用时 %0.3fs" % (time() - t0))
print(clf.best_estimator_) #打印最优估计
print("测试集SVM分类模型开始")
t0=time()
Y_pred=clf.predict(X_test_pca)
print("分类预测用时 %0.3fs" % (time() - t0))
print("误差衡量")
print(classification_report(Y_test,Y_pred,target_names=target_names))
print("预测值和实际值对角矩阵")
print(confusion_matrix(Y_test, Y_pred, labels=range(n_classes)))
#画图
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
   plt.figure(figsize=(1.8 * n_col, 2.4* n_row))
   plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
   for i in range(n_row * n_col):
       plt.subplot(n_row, n_col, i + 1)
       plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
       plt.title(titles[i], size=12)
       plt.xticks(())
       plt.yticks(())
# 绘制一部分测试集上的预测结果
def title(y_pred, y_test, target_names, i):
##以空格为分隔符，把y_pred分成一个list。分割的次数1。[-1]取最后一个
   pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
   true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
#   return '预测值: %s \n 真实值:%s' % (pred_name,true_name)
   return 'predicted: %s\ntrue:    %s' % (pred_name, true_name)
## 画人脸咯，eigenfaces主成分特征脸
eigenface_titles = ["eignfaces: %d" % i for i in range(eignfaces.shape[0])]
plot_gallery(eignfaces, eigenface_titles, h, w)
prediction_titles = [title(Y_pred, Y_test, target_names, i) for i in range(Y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w) 
plt.show()