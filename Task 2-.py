#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("movie_success_rate.csv")
df


# In[5]:


df.shape


# In[6]:


df.head()


# In[8]:


df.tail()


# In[10]:


df.columns


# In[12]:


df['Genre'].value_counts()


# In[13]:


df['Director'].value_counts()


# In[14]:


df['Actors'].value_counts()


# In[16]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[18]:


df = df.fillna(df.median())


# # LOGISTIC REGRESSION

# In[19]:


df.columns


# In[20]:


x = df[['Year',
       'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)',
       'Metascore', 'Action', 'Adventure', 'Aniimation', 'Biography', 'Comedy',
       'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War',
       'Western']]
y = df['Success']


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,stratify=y)


# In[23]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)


# In[24]:


log.score(x_test,y_test)


# In[25]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,log.predict(x_test))


# In[26]:


sns.heatmap(clf,annot=True)


# # SOME OPTIMAZTIONS

# In[27]:


#normalising all columns
x_train_opt = x_train.copy()
x_test_opt = x_test.copy()


# In[28]:


from sklearn.preprocessing import StandardScaler
x_train_opt = StandardScaler().fit_transform(x_train_opt)
x_test_opt = StandardScaler().fit_transform(x_test_opt)


# In[30]:


#fitting again in Logistic Regression
log.fit(x_train_opt,y_train)
log.fit(x_train_opt,y_train)


# In[32]:


log.score(x_test_opt,y_test)


# # KNN

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=40)
kn.fit(x_train,y_train)


# In[34]:


kn.score(x_test,y_test)


# # DECISION TREE 

# In[35]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree.score(x_test,y_test)


# In[36]:


tree.score(x_train,y_train)


# In[38]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,tree.predict(x_test))
clf


# In[39]:


sns.heatmap(clf,annot=True)


# In[ ]:




