# -*- coding: utf-8 -*-
# @Time    : 2017/2/24 20:22
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

#from Load_data import train_df, test_df, train_price
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

###交叉验证 选择最佳超参数
#1.1岭回归+CV   alpha=5.0 时达到最佳，score=0.93
# alphas = np.logspace(-3,2,50)
# test_scores=[]
# for alpha in alphas:
#     clf = linear_model.Ridge(alpha)  #使用岭回归
#     test_score = np.sqrt(model_selection.cross_val_score(clf,train_df, train_price['log_price+1'], cv=10))
#     test_scores.append(round(np.mean(test_score), 7))
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(alphas, test_scores, 'g-')
# plt.xlabel('alphas')
# plt.ylabel('scores')
# plt.show()



# #使用随机森林评估特征重要性
from load_data_new import train_df, test_df, train_price
# def random_forest_Regressor(train_x, train_y):
#     '随机森林分类器'
#     from sklearn.ensemble import RandomForestRegressor
#     model = RandomForestRegressor()
#     model.fit(train_x, train_y)
#     return model
#
# column_name = train_df.columns
# randomForest_model = random_forest_Regressor(train_df, train_price['log_price+1'])    #randomForest_model.feature_importances_  #越高说明特征越重要
# import pprint
# pprint.pprint(sorted(zip(map(lambda x: round(x, 3), randomForest_model.feature_importances_), column_name), reverse=True))
# #print sorted(zip(map(lambda x: round(x, 3), randomForest_model.feature_importances_), column_name), reverse=True)



##使用PCA  减少特征
test_df_Id = test_df[['Id']]
test_df = test_df.drop(['Id'], axis=1)

# from sklearn.decomposition import PCA
# pca=PCA(n_components=120, copy=True)  #保留的特征个数
# new_train_df = pca.fit_transform(train_df, train_price['log_price+1'])
# new_test_df = pca.transform(test_df)
#
# print pca.n_components  #保留的特征个数
# print pca.explained_variance_  #每个特征的方差
# print pca.explained_variance_ratio_  #每个特征各自占总方差的比例
# print sum(pca.explained_variance_ratio_)


new_train_df = train_df
new_test_df = test_df

# #随机森林 交叉验证    #选择max_features参数：1.0
# train_df = new_train_df
# max_features=[.1 ,.3 ,.5 ,.7 ,.9 ,.99]
# test_scores=[]
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators=50, max_features=max_feat)  #n_estimators 代表要多少棵树  最大特征个数、最大采样数量
#     test_score = np.sqrt(-model_selection.cross_val_score(clf, train_df, train_price['log_price+1'], cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#
# print test_scores
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(max_features, test_scores, 'g-')
# plt.xlabel('alphas')
# plt.ylabel('scores')
# plt.show()


# #弱学习器的最大迭代次数，或者说最大的弱学习器的个数，默认是10     选择n_estimators参数：8
# params = range(1, 81, 4)
# test_scores=[]
# for param in params:
#     clf = RandomForestRegressor(n_estimators=param,
#                                 max_features=1.0,
#                                 bootstrap=True,
#                                 oob_score=True,
#                                 )  #n_estimators 代表要多少棵树  最大特征个数、最大采样数量
#     test_score = np.sqrt(-model_selection.cross_val_score(clf, train_df, train_price['log_price+1'], cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(params, test_scores, 'g-')
# plt.xlabel('n_estimators')
# plt.ylabel('scores')
# plt.show()