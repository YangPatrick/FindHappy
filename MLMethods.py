#!/usr/bin/env python
# coding: utf-8

# In[4]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df_train = pd.read_csv("/Users/apple/Desktop/happiness_train_abbr.csv")
df_test = pd.read_csv("/Users/apple/Desktop/happiness_test_abbr.csv")


# In[6]:


df = pd.concat([df_train, df_test], axis = 0, sort = False)


# In[7]:


df.columns


# In[49]:


df.shape


# In[8]:


df.loc[df['happiness'].isna(), 'happiness'] = 4
df.loc[df['happiness'] == -8, 'happiness'] = 4
df.loc[df['family_income'] < 0, 'family_income'] = df['family_income'].mean()
df.loc[df['family_income'].isna(), 'family_income'] = df['family_income'].mean()


# In[9]:


df.groupby(by = 'happiness').count()


# In[10]:


df['family_income'].describe()


# In[11]:


df.isna().sum()/df.shape[0]


# In[12]:


X, y = df.drop(['id', 'happiness', 'survey_time','work_status', 'work_yr', 'work_type', 'work_manage'], axis = 1),df["happiness"]


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features = 4, random_state=0)
clf.fit(X_train, y_train)


# In[43]:


from sklearn.model_selection import GridSearchCV
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    rfc.fit(X_train, y_train)
    score = cross_val_score(rfc, X_test, y_test, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))

# 绘制学习曲线
x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()


# In[48]:


rfc = RandomForestClassifier(n_estimators = 25,
                             random_state = 90)
param_grid = {'max_depth':np.arange(1,20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(X_train, y_train)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


(clf.predict(X_test) == y_test).sum()/y_test.shape[0]


# In[16]:


clf.feature_importances_


# In[17]:


X.columns


# In[18]:


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train) 


# In[19]:


shap.summary_plot(shap_values, X_train, plot_type="bar")


# In[20]:


import eli5
from eli5.sklearn import PermutationImportance


# In[21]:


perm = PermutationImportance(clf, random_state = 0).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[22]:


importances = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(clf.feature_importances_,3)})
importances_df = importances.sort_values('Importance', ascending=False).reset_index(drop=True)
importances_plot = importances.sort_values('Importance', ascending=False).set_index('Feature')
importances_plot


# In[23]:


import plotly.graph_objects as go
fig_bar_feature = go.Figure()
fig_bar_feature.add_trace(go.Bar(x=importances_df['Feature'], y=importances_df['Importance']))
fig_bar_feature.update_layout(width=800, title_text='Bar chart representing Importances of Features',
                       xaxis_title_text='Features', yaxis_title_text='Importance', xaxis_tickangle=-45)
fig_bar_feature.show()


# In[52]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rfc, X_train, y_train, cv = 10, scoring='accuracy')


# In[55]:


np.round(scores, 3)


# In[27]:


df_combined = X.copy()


# In[28]:


for i in range(18, importances_df.shape[0]):
    column = importances_df['Feature'][i]
    df_combined.drop([column], inplace=True, axis=1)

df_combined.head()


# In[70]:


df_combined = df_combined.drop(['health_problem','health', 'status_3_before'], axis = 1)


# In[76]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_combined, y, test_size=0.3, random_state=42)


# In[78]:


clf = RandomForestClassifier()
model = clf.fit(X_train, y_train)


# In[81]:


y_pred_random_forest = model.predict(X_test)

random_forest_accuracy = round(model.score(X_train, y_train), 2)


# In[57]:


fig = plt.figure(figsize = (12, 12), dpi = 1000)
sns.heatmap(np.round(df_combined.corr(), 3), cmap="YlGnBu")
plt.savefig("corr.png")
plt.show()


# In[63]:


model = rfc.fit(X_train, y_train)


# In[68]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model = rfc, dataset = X_test, model_features = X_test.columns, feature = 'equity')

# plot it
pdp.pdp_plot(pdp_goals, 'equity')
plt.savefig('pdp.png')
plt.show()


# In[75]:


df_combined.columns


# In[ ]:




