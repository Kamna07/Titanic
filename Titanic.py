#!/usr/bin/env python
# coding: utf-8

# In[241]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')


# In[242]:


train = pd.read_csv(r'C:\Users\ony\Downloads\titanic\train.csv')
test = pd.read_csv(r'C:\Users\ony\Downloads\titanic\test.csv')


# In[243]:


y = test['PassengerId']
train.head(10)


# In[244]:


test.head(10)


# In[245]:


print(train.isnull().sum())
print(train.info())


# In[246]:


print(test.isnull().sum())
print(test.info())


# In[247]:


print(round(177/(len(train["Age"])),4))
ax = train['Age'].hist(bins = 15,color = 'teal',alpha = 0.8)
ax.set(xlabel = 'Age',ylabel = 'count')
plt.show()


# In[248]:


median = train['Age'].median()
train['Age'] = train['Age'].fillna(median)


# In[249]:


print(round(687/(len(train['Cabin'])),4))


# In[250]:


train.drop('Cabin', axis = 1,inplace = True)


# In[251]:


sns.countplot(x='Embarked',data = train,palette = 'Set2')
plt.show()


# In[252]:


train['Embarked'].fillna('S',inplace = True)
print(train.isnull().sum())


# In[253]:


mediantest = test['Age'].median()
test['Age'] = test['Age'].fillna(mediantest)

print(round(327/(len(test['Cabin'])),4))
test.drop('Cabin', axis = 1,inplace = True)

mediant1 = test['Fare'].median()
test['Fare'] = test['Fare'].fillna(mediant1)


# In[254]:


print(test.isnull().sum())


# In[255]:


trn1 = train

trn1.drop('PassengerId',axis = 1,inplace = True)
trn1.drop('Name',axis = 1,inplace = True)
trn1.drop('Ticket',axis = 1,inplace = True)

trn2 = pd.get_dummies(trn1,columns = ['Pclass'])
trn3 = pd.get_dummies(trn2,columns = ['Sex'])
trn4 = pd.get_dummies(trn3,columns = ['Embarked'])

trn = trn4

trn.head(5)


# In[256]:


tst1 = test

tst1.drop('PassengerId',axis = 1,inplace = True)
tst1.drop('Name',axis = 1,inplace = True)
tst1.drop('Ticket',axis = 1,inplace = True)

tst2 = pd.get_dummies(tst1,columns = ['Pclass'])
tst3 = pd.get_dummies(tst2,columns = ['Sex'])
tst4 = pd.get_dummies(tst3,columns = ['Embarked'])

tst = tst4

tst.head(5)


# In[ ]:





# In[257]:


'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0)
cls.fit(x_train,y_train)
pred_0 = cls.predict(x_test)
print(cls.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred_0)
print(cm)'''


# In[258]:


'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

from sklearn.svm import SVC
cls = SVC(kernel= 'linear',random_state = 0)
cls.fit(x_train,y_train)
pred_1 = cls.predict(x_test)
print(cls.score(x_test,y_test))'''


# In[259]:


'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

from sklearn.naive_bayes import GaussianNB 
cls = GaussianNB()
cls.fit(x_train, y_train)
pred_2 = cls.predict(x_test)
print(cls.score(x_test,y_test))'''


# In[260]:


x_train = trn.drop("Survived",axis=1)
y_train = trn["Survived"]
x_test  = tst

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(n_estimators = 300,criterion = 'entropy',max_depth = 20,random_state =0)
cls.fit(x_train,y_train)
pred = cls.predict(x_test)
print(cls.score(x_train,y_train))


# In[ ]:





# In[261]:


submission = pd.DataFrame({"PassengerId": y,"Survived": pred})

submission.to_csv('titanic.csv', index=False)

