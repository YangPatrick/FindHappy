#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import sys
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt  
import pandas as pd


# In[4]:


train = pd.read_csv('happiness_train_complete.csv',encoding='gbk')
train.head()


# In[7]:


#happiness均值为3.87，最大值为5，最小值为1，大部分人的幸福感是比较高的
train['happiness']=train['happiness'].replace(-8,3)
train.describe()


# In[21]:


#直方图函数定义
def distrib(col_name):
    col = train[col_name].dropna()
    plt.hist(col, 20)
    plt.title(col_name)
    plt.show()


# In[29]:


#盒图函数定义
def boxplot(col_name):
    fig, ax = plt.subplots(figsize=(3,4))
    ax.set_title(col_name)
    ax.boxplot(train[col_name].dropna())
    plt.show()


# In[30]:


distrib('happiness')
boxplot('happiness')


# In[20]:


#整体幸福状况
#五个等级饼图
f,ax=plt.subplots(1,2,figsize=(14,6))
explode=(0.05,0.05,0,0,0)
train['happiness'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,explode=explode)
ax[0].set_xlabel('happiness_level')
sns.countplot('happiness',data=train,ax=ax[1])
ax[1].set_title('How much people are happy',fontsize=15)
ax[1].set_xlabel('happiness_level')
ax[1].set_ylabel('number')
plt.show()


# In[38]:


#按性别分析
sns.countplot(x='gender',hue='happiness',data=train)
plt.title('The number of men and women at different levels of happiness',fontsize=15)
fig,ax1=plt.subplots(1,2,figsize=(10,5))
explode=(0.05,0.03,0,0,0)
train['happiness'][train['gender']==1].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[0],shadow=True,explode=explode)
train['happiness'][train['gender']==2].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[1],shadow=True,explode=explode)
fig.suptitle('Male and female happiness level pie chart',fontsize=18,verticalalignment='center')
#女性的幸福感略高于男性


# In[51]:


train['age']=2021-train['birth']
train['age']=pd.cut(train['age'],bins=[16,32,48,64,80,100],labels=['1','2','3','4','5'])
f,ax=plt.subplots(1,2,figsize=(14,6))
explode=(0.05,0.05,0,0,0)
train['age'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,explode=explode)
sns.countplot('age',data=train,ax=ax[1])
ax[1].set_xlabel('age range')
ax[1].set_ylabel('number')
plt.show()


# In[52]:


sns.countplot(x='age',hue='happiness',data=train)
plt.title('The number of people with different happiness levels in 5 ages',fontsize=15)


# In[59]:


fig,ax2=plt.subplots(1,5,figsize=(20,4))
explode=(0.05,0.03,0,0,0)
train['happiness'][train['age']=='1'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax2[0],shadow=True,explode=explode,title='16-32')
train['happiness'][train['age']=='2'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax2[1],shadow=True,explode=explode,title='32-48')
train['happiness'][train['age']=='3'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax2[2],shadow=True,explode=explode,title='48-64')
train['happiness'][train['age']=='4'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax2[3],shadow=True,explode=explode,title='64-80')
train['happiness'][train['age']=='5'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax2[4],shadow=True,explode=explode,title='80-')
fig.suptitle('The proportion of happiness level in different age',fontsize=18,verticalalignment='center')


# In[58]:


#按样本类型分析，1为城市，2为农村
f,ax=plt.subplots(1,3,figsize=(18,6))
sns.countplot(x='survey_type',hue='happiness',data=train,ax=ax[0])
explode=(0.05,0.03,0,0,0)
train['happiness'][train['survey_type']==1].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1],shadow=True,explode=explode,title='City')
train['happiness'][train['survey_type']==2].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[2],shadow=True,explode=explode,title='Rural area')
fig.suptitle('The number of people with different happiness level by survey_type',fontsize=18,verticalalignment='center')
        


# In[63]:


#按收入合理分析
fig,ax=plt.subplots(1,4,figsize=(20,4))
explode=(0.05,0.03,0,0,0)
train['happiness'][train['inc_ability']==1].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,explode=explode)
train['happiness'][train['inc_ability']==2].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1],shadow=True,explode=explode)
train['happiness'][train['inc_ability']==3].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[2],shadow=True,explode=explode)
train['happiness'][train['inc_ability']==4].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[3],shadow=True,explode=explode)
fig.suptitle('The proportion of happiness level in different inc_ability',fontsize=18,verticalalignment='center')


# In[61]:


#按公平分析
fig,ax=plt.subplots(1,5,figsize=(20,4))
explode=(0.05,0.03,0,0,0)
train['happiness'][train['equity']==1].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,explode=explode)
train['happiness'][train['equity']==2].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1],shadow=True,explode=explode)
train['happiness'][train['equity']==3].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[2],shadow=True,explode=explode)
train['happiness'][train['equity']==4].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[3],shadow=True,explode=explode)
train['happiness'][train['equity']==5].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[4],shadow=True,explode=explode)
fig.suptitle('The proportion of happiness level in different equity',fontsize=18,verticalalignment='center')


# In[ ]:




