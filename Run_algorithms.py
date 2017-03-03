# -*- coding: utf-8 -*-
# @Time    : 2017/2/24 20:23
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
#from Load_data import train_df, test_df, train_price, test_data_Id
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

###预测测试数据  用bagging
# bootstrap=True:使用“包外”样本计算范化误差  n_estimators:基学习器的数量  bootstrap=True:样本有放回抽取

#Ridge有问题
# clf = BaggingRegressor(base_estimator=linear_model.Ridge(alpha=5.0),
#                            n_estimators=50,
#                            bootstrap=True,
#                            oob_score=True).fit(train_df, train_price['log_price+1'])
# test_price_Ridge = clf.predict(test_df)
# test_price_Ridge =  np.expm1(test_price_Ridge)


# bagging + randomforest
from Feature_engineering import new_train_df, new_test_df, train_price, test_df_Id
# clf = BaggingRegressor(base_estimator=RandomForestRegressor(max_features=1.0,n_estimators=8),
#                            bootstrap=True,
#                            oob_score=True).fit(new_train_df, train_price['log_price+1'])
# test_price_Randomforest = clf.predict(new_test_df)
# test_price_Randomforest = np.expm1(test_price_Randomforest) #转换回正常价格
#
#
# #预测评分
# print clf.score(new_train_df, train_price['log_price+1'])   #该模型对于训练数据的得分
#
#
# test_df_Id['SalePrice'] = pd.DataFrame(test_price_Randomforest).apply(lambda x:np.round(x, 1))
# #test_df_Id.to_csv('result8.csv', index=False)  #生成结果文件
# print test_df_Id.head(10)



#模型用在训练集上   残差=真实值-预测值
# import matplotlib.pyplot as plt
# # plt.rcParams['figure.figsize'] = (6.0, 6.0)
# # preds = pd.DataFrame({"preds":clf.predict(new_train_df), "true":train_price['log_price+1']})
# # preds["residuals"] = preds["true"] - preds["preds"]
# # preds.plot(x = "preds", y = "residuals",kind = "scatter")
# # plt.show()
# #
# #
# #
# predictions = pd.DataFrame({"bagging":np.expm1(clf.predict(new_train_df)), "true":train_price['SalePrice']})
# predictions.plot(x = "bagging", y = "true", kind = "scatter")
# plt.ylim(0, )
# plt.show()




##使用 AdaBoostRegressor

#选出最优参数
from sklearn.ensemble import AdaBoostRegressor
params = range(1, 81, 4)
test_scores = []
# for param in params:    #n_estimators：15
#     clf = AdaBoostRegressor(base_estimator=RandomForestRegressor(max_features=0.75),
#                             n_estimators=param).fit(new_train_df, train_price['log_price+1'])
#
#     #预测评分
#     score = clf.score(new_train_df, train_price['log_price+1'])   #该模型对于训练数据的得分
#     #score = np.corrcoef(test_price_Randomforest, train_price['SalePrice'])
#     test_scores.append(score)
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(params, test_scores, 'g-')
# plt.xlabel('n_estimators')
# plt.ylabel('scores')
# plt.show()

#使用AdaBoostRegressor来预测房价
clf = AdaBoostRegressor(base_estimator=RandomForestRegressor(max_features=1.0),
                        n_estimators=15).fit(new_train_df, train_price['log_price+1'])
test_price_Randomforest = clf.predict(new_test_df)
test_price_Randomforest = np.expm1(test_price_Randomforest) #转换回正常价格


#预测评分
print clf.score(new_train_df, train_price['log_price+1'])   #该模型对于训练数据的得分

test_df_Id['SalePrice'] = pd.DataFrame(test_price_Randomforest).apply(lambda x:np.round(x, 3))
#test_df_Id.to_csv('result10.csv', index=False)  #生成结果文件
print test_df_Id.head(10)

import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (6.0, 6.0)
# preds = pd.DataFrame({"preds":clf.predict(new_train_df), "true":train_price['log_price+1']})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")
# plt.show()

predictions = pd.DataFrame({"bagging":np.expm1(clf.predict(new_train_df)), "true":train_price['SalePrice']})
predictions.plot(x = "bagging", y = "true", kind = "scatter")
plt.xlabel(0)
plt.ylim(0)
plt.show()