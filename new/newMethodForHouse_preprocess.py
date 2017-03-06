# -*- coding: utf-8 -*-
# @Time    : 2017/3/5 10:46
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
from scipy.stats import skew

TARGET = 'SalePrice'

## Load the data ##
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

ntrain = train.shape[0] #训练集数据个数
ntest = test.shape[0]  #测试集数据个数

## Preprocessing ##
y_train = np.log(train[TARGET]+1)  #对训练集的房价进行对数转换
#print y_train[:5]

train.drop([TARGET], axis=1, inplace=True) #训练集删除价格一栏

#训练集和测试集合并 方便后续处理
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
print '原始all_data：', all_data.shape

#log transform skewed numeric features:  找出那些数值型的特征
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index  #特征名

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #计算该特征的偏斜度
skewed_feats = skewed_feats[skewed_feats > 0.75]  #过滤掉偏斜度>0.75的特征
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])  #对数值型特征进行对数转换

all_data = pd.get_dummies(all_data) #进行one-hot编码

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
print '预处理+one-hot后的all_data：', all_data.shape

#creating matrices for sklearn:
#重新拆开为训练集和测试集
x_train = np.array(all_data[:train.shape[0]])
x_test = np.array(all_data[train.shape[0]:])
print '从总数据拆分后的x_train, x_test：', x_train.shape, x_test.shape