# -*- coding: utf-8 -*-
# @Time    : 2017/2/23 10:58
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

'''
Kaggle——预测房价
'''
import pandas as pd
import numpy as np

################################### 加载训练集和测试集数据 ################################
train_data = pd.read_csv('./data/train.csv')

test_data = pd.read_csv('./data/test.csv')
test_data['SalePrice'] = 0.0 #只是方便后面和训练集合并不出现NA值


####################  train.data：(1460, 81)   test_data:(1459, 81)


######################################## 数据预处理 ##########################################

##1、处理训练集
# #删除指定列名的数据 删除前18个缺失数据量最大的列   (2919,62)
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
train_data = train_data.drop(missing_data.index[:18], axis=1)


#去掉'Electrical'这列中缺失的一行(训练集)
train_data = train_data.drop(train_data.loc[ train_data['Electrical'].isnull() ].index)

# #去掉'GrLivArea'中两个异常值 (训练集)
# print train_data.sort_values(by = 'GrLivArea', ascending = False)['GrLivArea'][:5]  #由观察图像得出删除两个异常值
train_data = train_data.drop([1298,523])

########################################################## train.data：(1457, 63)

##2、处理测试集
# #删除指定列名的数据 删除前18个缺失数据量最大的列   (2919,62)
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
test_data = test_data.drop(missing_data.index[:18], axis=1)

# # #找出数据集中有空值的特征 在空值的位置用该列的众数填充
# #方法1：
# NA_count = test_data.isnull().sum().sort_values(ascending=False)  #返回一个Series
# NA_count_columnname = NA_count[NA_count > 0].index  #有空值的特征名
# #print NA_count[NA_count > 0] #返回一个Series
# most_filling = test_data[NA_count_columnname].dropna().mode().values  #获取每个有空值的特征的众数  列表形式
# for i in range(len(NA_count_columnname)):
#     test_data.loc[(test_data[NA_count_columnname[i]].isnull()), NA_count_columnname[i]] = most_filling[0][i]

# #方法2：
column_most =  test_data.iloc[:,1:].dropna().mode().iloc[0]  #找出每列数据统计最多的一个
test_data.fillna(column_most, inplace=True)       #以每个特征的众数填充各自的NAN值

####################################################################### test_data：(1459, 63)



# # 合并训练集和测试集  对所有分类变量用 one-hot 编码     (对于分类型变量，更好的方法是one-hot编码)
train_data = train_data.reset_index(drop=True)  #因为训练集删除了一些行，需要重新建立索引

all_df = pd.concat((train_data, test_data), axis=0)  #(2916, 63)
all_dummy_df = pd.get_dummies(all_df)   #(2916, 221)
#print all_dummy_df.isnull().sum().sort_values(ascending=False)

# # #标准化数据集  只需对原来是数值型的特征进行标准化
from sklearn import preprocessing
numeric_colnum = all_dummy_df.columns[all_dummy_df.dtypes != 'object'][1:]  #找出原来哪些特征是数值型 去掉ID这列
numeric_colnum = numeric_colnum.drop('SalePrice')
data_scale = preprocessing.scale(all_dummy_df[numeric_colnum], axis=0)
all_dummy_df.loc[:,numeric_colnum] = data_scale



# # # #重新拆分成训练集和测试集
train_df=all_dummy_df.iloc[0:train_data.shape[0], :]  #训练集
test_df = all_dummy_df.iloc[train_data.shape[0]:, :]  #测试集
train_price = train_df[['SalePrice']];train_price['log_price+1'] = np.log1p(train_df['SalePrice'])  #log(x+1) 转为符合正态分布

#去掉训练集和测试集中的 'SalePrice' 一列
train_df = train_df.drop(['Id','SalePrice'], axis=1)
test_df = test_df.drop(['SalePrice'], axis=1)

###########################此时：tain.df(1457,221) test.df(1459,220)   预处理和标准化后的训练集和测试集 ###############


###
# 该文件说明：
# train_price:训练数据的房价 一列为真实房价，一列为正态转后的房价
# train_df：预处理和标准化后的训练集 包含'Id'和'Saleprice'一栏
# test_df：预处理和标准化后的测试集  不包含'Saleprice'一栏
###


# print train_df.shape
# print test_df.shape