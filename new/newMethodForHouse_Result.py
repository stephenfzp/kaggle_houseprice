# -*- coding: utf-8 -*-
# @Time    : 2017/3/5 10:56
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

from newMethodForHouse_preprocess import x_train, x_test, y_train, ntrain, ntest
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt

TARGET = 'SalePrice'
NFOLDS = 4
SEED = 0
NROWS = None
SUBMISSION_FILE = '../data/sample_submission.csv'

#根据训练集设置一个KFold交叉验证
kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)  # 根据训练数据的根数  使用4次分层


class SklearnWrapper(object):
    'ExtraTreesRegressor RandomForestRegressor Ridge Lasso的函数参数一样 可以用这个模板复用'
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        print clf

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    'xgboost的模板'
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)
        print xgb

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf): #传入一个model
    oof_train = np.zeros((ntrain,))  #新建一个长度=原训练集行数的0矩阵  用来存放训练集拟合的结果
    oof_test = np.zeros((ntest,))    #新建一个长度=原测试集行数的0矩阵  用来存放测试集拟合的结果
    oof_test_skf = np.empty((NFOLDS, ntest)) #未初始化的随机值矩阵 行=NFOLDS 列=测试集数据个数

    for i, (train_index, test_index) in enumerate(kf): #迭代次数为NFOLDS次
        #train_index，test_index分别为KFold在原训练集中拆分出来的小训练集和小测试集索引
        x_tr = x_train[train_index]  #分配到的训练数据
        y_tr = y_train[train_index]  #训练数据中对应的房价
        x_te = x_train[test_index]   #训练集中分配的测试集

        clf.train(x_tr, y_tr) #根据给定分配出来的训练数据和房价训练模型

        #因为每次分层训练集和测试集的索引都不同 当分层结束 每次分层的索引加起来可以刚好是一个完整的数据集的索引
        #oof_train中原是一个0矩阵，但每次分层就会有一定的索引位置会被填充，分层结束整个矩阵就刚好被填充完毕
        oof_train[test_index] = clf.predict(x_te) #根据训练好的模型来测试从训练集中划分出来的测试集
        oof_test_skf[i, :] = clf.predict(x_test)  #根据训练好的模型来测试全部测试集
    # oof_test_skf中保存了该模型跑了NFOLDS次完整测试集的结果 求均值保存到oof_test中
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) #列表中的每个值都用列表包装好

# 5个分类器的设置参数
#ExtraTreesRegressor参数
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

#Randomforest参数
rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

#Xgboost参数
xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}

#Ridge参数
rd_params={
    'alpha': 10
}

#Lasso参数
ls_params={
    'alpha': 0.005
}


xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)  #train：(1460L, 1L)  test：(1459L, 1L)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

# mean_squared_error():#计算均方根误差 数值越小表示精确度更高
print '\n'
print("Xgboost-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ExtraTreesRegressor-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RandomForestRegressor-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("Ridge-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("Lasso-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
print '\n'

#按索引拼接 索引相同 右横向拼接
x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)


#最后训练集和测试集列数被浓缩成几列 每个弱分类器的结果占一列
print("{},{}".format(x_train.shape, x_test.shape))
print '\n'


pd.DataFrame(x_train).to_csv('x_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(x_test).to_csv('x_test.csv', index=False)


