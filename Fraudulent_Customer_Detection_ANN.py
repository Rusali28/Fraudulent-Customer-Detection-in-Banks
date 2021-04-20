#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[2]:


dataframe = pd.read_csv("C:/Users/rishu/OneDrive/Desktop/project/Rusali/Fraudulent-Customer-Detection-in-Banks-main/Churn_Modelling.csv")


# In[3]:


dataframe.head()


# In[4]:


dataframe.columns


# In[5]:


dataframe.index


# In[6]:


dataframe.info()


# In[7]:


X = dataframe.iloc[:, 3:13].values


# In[8]:


Y = dataframe.iloc[:, -1].values


# In[9]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[10]:


labelE_X1 = LabelEncoder()


# In[11]:


X[:,1] = labelE_X1.fit_transform(X[:,1])


# In[12]:


labelE_X2 = LabelEncoder()
X[:,2] = labelE_X1.fit_transform(X[:,2])


# In[13]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[14]:


X = X[:,1:]


# In[15]:


X.shape


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 40)


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[19]:


import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[20]:


classifier = Sequential()


# In[21]:


classifier.add(Dense(units = 9, input_shape=(11,),activation =  'relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units = 4, activation =  'relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units = 1, activation =  'sigmoid'))


# In[22]:


classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[23]:


classifier.fit(X_train, Y_train, batch_size=25, epochs=200)


# In[24]:


y_pred = classifier.predict(X_test)


# In[25]:


y_pred


# In[26]:


y_pred = (y_pred>0.5)


# In[27]:


y_pred


# In[28]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[29]:


pred = confusion_matrix(Y_test, y_pred)


# In[30]:


pred


# In[31]:


accuracy_score(Y_test, y_pred)


# In[32]:


from keras.wrappers.scikit_learn import KerasClassifier


# In[33]:


from sklearn.model_selection import GridSearchCV


# In[34]:


def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 9, input_shape=(11,), activation =  'relu'))
    classifier.add(Dense(units = 4, activation =  'relu'))
    classifier.add(Dense(units = 1, activation =  'sigmoid'))
    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier


# In[35]:


classifier = KerasClassifier(build_fn = build_classifier)


# In[36]:


parameters = {'batch_size': [ 25,35], 'epochs' : [200,300], 'optimizer':['adam','rmsprop']}


# In[37]:


grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy',cv=10)


# In[38]:


grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[39]:


best_parameters


# In[42]:


classifier1 = Sequential()


# In[43]:


classifier1.add(Dense(units = 9, input_shape=(11,),activation =  'relu'))
classifier1.add(Dropout(rate=0.1))
classifier1.add(Dense(units = 4, activation =  'relu'))
classifier1.add(Dropout(rate=0.1))
classifier1.add(Dense(units = 1, activation =  'sigmoid'))


# In[45]:


classifier1.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[46]:


classifier1.fit(X_train, Y_train, batch_size=25, epochs=300)


# In[48]:


y_pred1 = classifier1.predict(X_test)


# In[49]:


y_pred1


# In[50]:


y_pred1 = (y_pred1>0.5)


# In[51]:


pred1 = confusion_matrix(Y_test, y_pred1)


# In[52]:


pred1


# In[54]:


accuracy_score(Y_test, y_pred1)


# In[ ]:




