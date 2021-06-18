import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train=pd.read_csv('happiness_train_complete.csv',encoding='gbk')
df_test=pd.read_csv('happiness_test_complete.csv',encoding='gbk')
df_train.info()

#查看label
df_train=df_train[df_train['happiness']>0]
df_train['happiness'].plot.hist()
#提取label
y_train=np.log1p(df_train.pop('happiness'))
#合并数据
df_all=pd.concat((df_train,df_test),axis=0)
#1.处理缺失值
#查看缺失值的列
total = df_all.isnull().sum().sort_values(ascending=False)
percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=['total', 'percent'])
missing_data[missing_data['percent']>0]
#配偶状况s_work_exper,s_hukou,s_political,s_birth,s_edu,s_income
marital_data=df_all[df_all['marital']==1]
#缺失值基本都是由于未婚造成的
#全部缺失值填充为0
df_all['s_work_exper'].fillna(0,inplace=True)
df_all['s_hukou'].fillna(0,inplace=True)
df_all['s_political'].fillna(0,inplace=True) #删除
df_all['s_birth'].fillna(0,inplace=True)    #删除
df_all['s_edu'].fillna(0,inplace=True)     
df_all['s_income'].fillna(0,inplace=True)     #保留
df_all['s_work_status'].fillna(0,inplace=True) 
df_all['s_work_type'].fillna(0,inplace=True) 
edu_data=df_train[df_train['edu']==1]
edu_data['edu_status']
#缺失值是由于没有受过教育造成的
#全部缺失值填充为0
df_all['edu_status'].fillna(0,inplace=True)
df_all[df_all['edu_yr'].isnull()]
#查看social_friend和social_neighbor的缺失值
social_data=df_train[df_train['socialize']==1]
social_data['social_friend']
#缺失值是由于社交不频繁造成的
#全部缺失值填充为7
df_all['social_friend'].fillna(7,inplace=True)
df_all['social_neighbor'].fillna(7,inplace=True)
#全部填充为0,因为没有孩子
df_all['minor_child'].fillna(0,inplace=True)
#户口情况hukou_loc
hukou_data=df_train[df_train['hukou_loc'].isnull()]
hukou_data['hukou']
#缺失值是由于没有户口造成的
#全部缺失值填充为4
df_all['hukou_loc'].fillna(4,inplace=True)
df_all['work_type'].fillna(0,inplace=True)
df_all['work_status'].fillna(0,inplace=True)
df_all['work_manage'].fillna(0,inplace=True)
df_all[df_all['work_exper']==4|6]['work_yr']
df_all['work_yr'].fillna(0,inplace=True)
df_all['marital_now'].fillna(2015,inplace=True) 
#新增特征值mar_yr
df_all['mar_yr']=2015-df_all['marital_now']
#查看一下测试集的family_income有没有缺失值，发现没有
df_test['family_income'].isnull().value_counts()
#训练集中的family_income只有一条空记录，用平均值填充
df_all['family_income'].fillna(df_all['family_income'].mean(),inplace=True)
#缺失比例大的特征，其比例已经超过了60%，我们直接删除
df_all.drop( ['edu_other','invest_other','property_other','join_party',
                'edu_yr','marital_1st','s_political',],axis=1,inplace=True)
#验证处理结果
total = df_all.isnull().sum().sort_values(ascending=False)
percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
missing_data[missing_data['percent']>0]

#转换时间格式,求出每个人的年龄
df_all['age']=pd.to_datetime(df_all['survey_time']).dt.year-df_all['birth']
df_all.drop(['survey_time','birth'],axis=1,inplace=True)
#删除其他时间数据
df_all.drop(['s_birth','f_birth','m_birth'],axis=1,inplace=True)
df_all.drop(['marital_now'],axis=1,inplace=True)
#对于数值型数据都要进行标准化处理，这样做的目的时减小方差，提高迭代速度，同时也能提高精度
numeric_cols=['income','height_cm','weight_jin','s_income',
              'family_income','family_m','house','car'
              ,'son','daughter','minor_child','inc_exp','public_service_1',
              'public_service_2','public_service_3','public_service_4',
              'public_service_5','public_service_6','public_service_7',
              'public_service_8','public_service_9','floor_area']
numeric_cols_means=df_all.loc[:,numeric_cols].mean()
numeric_cols_std=df_all.loc[:,numeric_cols].std()
df_numeric=(df_all.loc[:,numeric_cols]-numeric_cols_means)/numeric_cols_std
df_numeric.iloc[:,1].hist()
#对类别类型的特征值转换为Object
df_object=df_all.drop(numeric_cols,axis=1)
df_object=df_object.astype(str)
for cols in list(df_object.iloc[:,1:].columns):
    df_object=pd.get_dummies(df_object.iloc[:,1:],prefix=cols)
#合并数据 
data=pd.concat((df_object,df_numeric),axis=1)