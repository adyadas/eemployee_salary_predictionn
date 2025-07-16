#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Employee Salary Prediction using adultcsv
#load your library
import pandas as pd


# In[3]:


data=pd.read_csv(r"C:\Users\Adya\Downloads\adult 3.csv")


# In[4]:


data


# In[5]:


data.tail()


# In[6]:


#Null Values
data.isna()


# In[7]:


data.isna().sum()


# In[8]:


print(data.occupation.value_counts())


# In[9]:


print(data.gender.value_counts())


# In[10]:


print(data['marital-status'].value_counts())


# In[11]:


print(data['education'].value_counts())


# In[12]:


print(data['workclass'].value_counts())


# In[14]:


print(data.occupation.replace({'?':'others'},inplace=True))


# In[18]:


data.workclass.replace({'?':'NotListed'},inplace=True)


# In[19]:


print(data['workclass'].value_counts())


# In[20]:


data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']


# In[21]:


print(data['workclass'].value_counts())


# In[22]:


data.shape


# In[23]:


data=data[data['education']!='5th-6th']
data=data[data['education']!='1th-4th']
data=data[data['education']!='Preschool']


# In[24]:


print(data['education'].value_counts())


# In[25]:


data.shape


# In[27]:


#redundancy
data.drop(columns=['education'],inplace=True)


# In[28]:


data


# In[29]:


#outlier
import matplotlib.pyplot as plt
plt.boxplot(data['age'])
plt.show()


# In[30]:


data=data[(data['age']<=75)&(data['age']>=17)]


# In[31]:


plt.boxplot(data['age'])
plt.show()


# In[32]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])
data


# In[34]:


x=data.drop(columns=['income'])
y=data['income']


# In[35]:


x


# In[36]:


y


# In[37]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x


# In[38]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y, test_size=0.2, random_state=23, stratify=y)


# In[39]:


xtrain


# In[40]:


#machine learning algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain, ytrain)    #input and output training   my pictures,  sad and happy
predict=knn.predict(xtest)                              pictures to predict sad or happy
predict #predicted value


# In[43]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, predict)


# In[45]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain, ytrain) #inputand output training
predict1=lr.predict(xtest)
predict1


# In[46]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict1)


# In[47]:


from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='adam',hidden_layer_sizes=(5,2), random_state=2, max_iter=2000)
clf.fix(xtrain, ytrain)
predict2=clf.predict(xtest)
predict2


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, predict2)


# In[53]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

models ={
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
    
}

results={}
for name, model in models.items():
    pipe = Pipeline ([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_train)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))






# In[ ]:




