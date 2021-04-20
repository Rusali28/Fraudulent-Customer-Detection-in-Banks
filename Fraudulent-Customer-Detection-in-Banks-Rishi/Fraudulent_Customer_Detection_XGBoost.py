#!/usr/bin/env python
# coding: utf-8

# In[28]:


import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[29]:


dataframe = pd.read_csv("C:/Users/rishu/OneDrive/Desktop/project/Rusali/Fraudulent-Customer-Detection-in-Banks-Rishi/Churn_Modelling.csv")


# In[30]:


dataframe.head()


# In[31]:


dataframe.columns


# In[32]:


dataframe.index


# In[33]:


dataframe.info()


# In[34]:


X = dataframe.iloc[:, 3:13].values


# In[35]:


Y = dataframe.iloc[:, -1].values


# In[36]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[37]:


labelE_X1 = LabelEncoder()
X[:,1] = labelE_X1.fit_transform(X[:,1])


# In[38]:


labelE_X2 = LabelEncoder()
X[:,2] = labelE_X1.fit_transform(X[:,2])


# In[39]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[40]:


X = X[:,1:]


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 40)


# In[42]:


from sklearn.preprocessing import StandardScaler


# In[43]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[44]:


import xgboost


# In[45]:


model = xgboost.XGBClassifier(random_state=40)


# In[46]:


model.fit(X_train,Y_train)


# In[47]:


y_pred = model.predict(X_test)


# In[48]:


y_pred


# In[49]:


from sklearn.metrics import accuracy_score


# In[50]:


accuracy_score(Y_test, y_pred)


# ### Parameter Tuning for XGBoost

# In[54]:


parameters = {
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "learning_rate": [0.5, 0.10, 0.15, 0.20, 0.25, 0.30],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}


# In[55]:


from sklearn.model_selection import RandomizedSearchCV


# In[56]:


rcv = RandomizedSearchCV(model, param_distributions=parameters,n_iter=5,scoring="roc_auc", n_jobs=1, cv=5, verbose=3)


# In[58]:


rcv.fit(X_train,Y_train)


# In[59]:


rcv.best_params_


# In[66]:


model_new = xgboost.XGBClassifier(min_child_weight=1,max_depth=5,learning_rate=0.1,gamma=0.4,colsample_bytree=0.5)


# In[67]:


model_new.fit(X_train,Y_train)


# In[68]:


y_pred_new = model_new.predict(X_test)


# In[69]:


y_pred_new


# In[70]:


accuracy_score(Y_test, y_pred_new)

