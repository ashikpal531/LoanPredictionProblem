# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 02:09:55 2020

@author: acer
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

#Reading data
train=pd.read_csv("train_ctrUa4K.csv")
test=pd.read_csv("test_lAUu6dG.csv")

#making a copy of original datasets
train_original=train.copy()
test_original=test.copy()

#print(train.columns)
#print(test.columns)

print("train: ",train.shape)
print("test: ",test.shape)

#Visualizing target variable
#print(train['Loan_Status'].value_counts())
#print(train['Loan_Status'].value_counts().plot.bar())

#Visualizing categorical features
'''plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.show()'''
#Viualizing ordinal features
'''print(train.columns)
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize= True).plot.bar(figsize=(24,6), title='Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()'''

#Visualizing numerical features
'''plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()
train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")
Text(0.5,0.98,'')
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()'''

#Imputation

#print(train.isnull().sum())

train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)

#print(train['Loan_Amount_Term'].value_counts())

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)
#print(train.isnull().sum())


test['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
test['Married'].fillna(train['Married'].mode()[0],inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)

#print(test.isnull().sum())

#print(train['LoanAmount'].hist(bins=20))
#print(test['LoanAmount'].hist(bins=20))

train['LoanAmount_log']=np.log(train['LoanAmount'])
#print(train['LoanAmount_log'].hist(bins=20))
test['LoanAmount_log']=np.log(test['LoanAmount'])

#Feature Engineering

train['Total_Income']= train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']

#sns.distplot(train['Total_Income'])

train['Total_Income_log'] = np.log(train['Total_Income'])
#sns.distplot(train['Total_Income_log']) 
test['Total_Income_log'] = np.log(test['Total_Income'])

train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
#sns.distplot(train['EMI']);

train['Balance Income']=train['Total_Income']-(train['EMI']*1000) 
# Multiply with 1000 to make the units equal 
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)
#sns.distplot(train['Balance Income']);

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 
                  'LoanAmount', 'Loan_Amount_Term'], axis=1) 
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 
                'LoanAmount', 'Loan_Amount_Term'], axis=1)
    
#Model building
    
#x = train.drop('Loan_Status',1) 
#y = train.Loan_Status 
#Logistic Regression
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

x=train.drop('Loan_Status',1)
y=train.Loan_Status
#Target variable is saved in seperate dataset

x=pd.get_dummies(x)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression()
model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling =1, max_iter=100,multi_class='ovr', n_jobs=1,
                   penalty='l2',random_state=1,solver='liblinear',tol=0.0001,verbose=0,warm_start=False)

pred_train= model.predict(x_train)
print("Logistic Regression: ")
print("TrainAccuracy: ",accuracy_score(y_train,pred_train))

pred_test=model.predict(x_test)
A_lr=print("TestAccuracy: ",accuracy_score(y_test,pred_test))

'''#Stratified k-fold cross validation
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(x,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = x.loc[train_index],x.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]
         
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
pred_test = model.predict(test) 
pred=model.predict_proba(xvl)[:,1]'''

#Decision Tree
print("Decision Tree: ")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
dt.fit(x_train,y_train)
pred_test_dt=dt.predict(x_test)
A_dt=print("Accuracy: ",accuracy_score(y_test,pred_test_dt))

#Random Forest
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
forest= RandomForestClassifier(criterion='entropy',n_estimators=10,
                               random_state=1,n_jobs=2)
forest.fit(x_train,y_train)
y_pred=forest.predict(x_test)
print("RandomForest: ")
#print('misclassifiedsamples:%d'%(y_test!=y_pred).sum()) #compute
A_rf=print('Accuracy:%.2f'%accuracy_score(y_test,y_pred))

#Ensemble Learning
print("Ensemble Learning: ")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf1= LogisticRegression()#penalty='l2',C=0.001,random_state=1)
clf2= DecisionTreeClassifier()#max_depth=1,criterion='entropy',random_state=0)
clf3= RandomForestClassifier()

   #Majority Voting Classifier
from sklearn.ensemble import VotingClassifier
mv_clf=VotingClassifier(estimators=[('lr',clf1),('dt',clf2),('rf',clf3)])
mv_clf=mv_clf.fit(x_train,y_train)
pred=mv_clf.predict(x_test)
from sklearn.metrics import accuracy_score
A_el=print("Accuracy: ",accuracy_score(pred,y_test))

#XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50, max_depth=4)     
model.fit(x_train, y_train)     
pred_test = model.predict(x_test)     
score = accuracy_score(y_test,pred_test)     
print('accuracy_score',score)     










