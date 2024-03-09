#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np


# In[56]:


data=pd.read_csv("data[1].csv")


# In[57]:


display(data.dtypes)





# In[59]:


data['bedrooms'] = data['bedrooms'].astype(int)


# In[60]:


data['bathrooms'] = data['bathrooms'].astype(int)


# In[61]:


display(data.dtypes)





# In[10]:


data.head()


# In[62]:


data.info()


# In[12]:


data.shape


# In[63]:


for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)


# In[64]:


data.isna().sum()


# In[65]:


data.describe()


# In[66]:


data.info()


# In[67]:


data['bedrooms'].value_counts()


# In[68]:


data.head()


# In[69]:


data['price_per_sqft']=data['price']*100000/data['sqft_lot']


# In[70]:


data['price_per_sqft']


# In[71]:


data.describe()


# In[22]:


data.shape


# In[72]:


data


# In[73]:


data.drop(columns=['waterfront'],inplace=True)


# In[74]:


data.head()


# In[76]:


data.drop(columns=['street'],inplace=True)


# In[77]:


data.drop(columns=['city'],inplace=True)


# In[78]:


data.drop(columns=['country'],inplace=True)


# In[79]:


data.drop(columns=['statezip'],inplace=True)


# In[80]:


data.drop(columns=['date'],inplace=True)


# In[95]:


data.drop(columns=['yr_built'],inplace=True)


# In[97]:


data.drop(columns=['yr_renovated'],inplace=True)


# In[120]:


data.drop(columns=['view'],inplace=True)


# In[122]:


data.drop(columns=['condition'],inplace=True)


# In[123]:


data.head()


# In[124]:


data.to_csv("final_dataset.csv")


# In[129]:


x=data.drop(columns=['price'])
y=data['price']


# In[146]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[147]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)


# In[148]:


print(x_train.shape)
print(y_train.shape)


# In[37]:


#applying linear regression


# In[149]:


column_trans=make_column_transformer((OneHotEncoder(sparse=False), ['bedrooms']), remainder='passthrough')


# In[151]:


scaler=MinMaxScaler()


# In[135]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[136]:


data.head()


# In[152]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)

lr=LinearRegression()
lr.fit(x_scaled, y)


# In[153]:


pipe=make_pipeline(column_trans,scaler, lr)


# In[154]:


pipe.fit(x_train,y_train)


# In[155]:


data.head()


# In[141]:


data.head(12)


# In[156]:


pipe.fit(x_train, y_train)


# In[157]:


y_pred_lr = pipe.predict(x_test)
print(y_pred_lr)


# In[118]:


r2_score(y_test,y_pred_lr)


# In[47]:


lasso=Lasso()


# In[48]:


pipe=make_pipeline(column_trans,scaler,lasso)


# In[49]:


pipe.fit(X_train,y_train)


# In[50]:


y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)


# In[51]:


ridge=Ridge()


# In[52]:


pipe=make_pipeline(column_trans,scaler,ridge)


# In[53]:


pipe.fit(X_train,y_train)


# In[54]:


y_pred_ridge=pipe.predict(X_test)
r2-score(y_test,y_pred_ridge)

