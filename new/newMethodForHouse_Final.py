# -*- coding: utf-8 -*-
# @Time    : 3/4/17 11:28 PM
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

#from newMethodForHouse_Result import y_train, SUBMISSION_FILE
import pandas as pd
import numpy as np
import xgboost as xgb

TARGET = 'SalePrice'
SUBMISSION_FILE = '../data/sample_submission.csv'

NFOLDS = 4
SEED = 0
NROWS = None

x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('x_test.csv')


# #xgboost加载的数据存储在对象DMatrix中
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

#使用xgboost的cv验证 return:evaluation history（list(string)）
res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=1000, verbose_eval=10, show_stdv=True)
print
print res.shape, type(res)
print

print res.tail()

best_nrounds = res.shape[0] - 1  #模型的训练迭代次数
cv_mean = res.iloc[-1, 0]  #测试集的标准方差平均值
cv_std = res.iloc[-1, 1]   #测试集的标准方差中位数

print('Ensemble-CV: {0} + {1}'.format(cv_mean, cv_std))
print '\n'

gbdt = xgb.train(xgb_params, dtrain, best_nrounds) #使用之前5个分类器的输出作为xgboost的输入

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)  #预测测试集
saleprice = np.exp(submission['SalePrice'])-1
submission['SalePrice'] = saleprice
#submission.to_csv('xgstacker_starter.sub.csv', index=None)
print submission.head()

