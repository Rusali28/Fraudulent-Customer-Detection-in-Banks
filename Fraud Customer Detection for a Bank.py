#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf


# In[10]:


import numpy as np
import pandas as pd


# In[11]:


dataframe = pd.read_csv("Churn_Modelling.csv")


# In[12]:


dataframe.head()


# In[13]:


dataframe.columns


# In[14]:


dataframe.index


# In[15]:


dataframe.info()


# In[16]:


X = dataframe.iloc[:, 3:13].values


# In[17]:


Y = dataframe.iloc[:, -1].values


# In[18]:


X


# In[19]:


Y


# In[20]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[21]:


labelE_X1 = LabelEncoder()


# In[22]:


X[:,1] = labelE_X1.fit_transform(X[:,1])


# In[23]:


X


# In[24]:


labelE_X2 = LabelEncoder()
X[:,2] = labelE_X1.fit_transform(X[:,2])


# In[25]:


X


# In[26]:


Y


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[28]:


X


# In[29]:


X = X[:,1:]


# In[30]:


X


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


sc = StandardScaler()


# In[34]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ## Building the ANN

# In[35]:


import keras


# In[36]:


from tensorflow.keras.models import Sequential


# In[37]:


from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[38]:


classifier = Sequential()


# In[39]:


classifier.add(Dense(units = 6, activation =  'relu'))
classifier.add(Dropout(rate=0.1))


# In[40]:


classifier.add(Dense(units = 6, activation =  'relu'))
classifier.add(Dropout(rate=0.1))


# In[41]:


classifier.add(Dense(units = 1, activation =  'sigmoid'))


# In[42]:


classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[35]:


classifier.fit(X_train, Y_train, batch_size=10, epochs=100)


# In[36]:


y_pred = classifier.predict(X_test)


# In[37]:


y_pred


# In[38]:


y_pred = (y_pred>0.5)


# In[39]:


y_pred


# In[40]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[41]:


pred = confusion_matrix(Y_test, y_pred)


# In[42]:


pred


# In[43]:


accuracy_score(Y_test, y_pred)


# ## Hyperparameter tuning

# In[43]:


from keras.wrappers.scikit_learn import KerasClassifier


# In[44]:


from sklearn.model_selection import GridSearchCV


# In[45]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[56]:


def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation =  'relu'))
    classifier.add(Dense(units = 6, activation =  'relu'))
    classifier.add(Dense(units = 1, activation =  'sigmoid'))
    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier


# In[57]:


classifier = KerasClassifier(build_fn = build_classifier)


# In[61]:


parameters = {'batch_size': [25, 32], 'epochs' : [100,500], 'optimizer':['adam','rmsprop']}


# In[62]:


grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)


# In[63]:


grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[64]:


best_accuracy


# In[65]:


best_parameters


# In[ ]:




