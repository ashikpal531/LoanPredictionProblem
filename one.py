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
#train_original=train.copy()
#test_original=test.copy()

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
plt.show()'''
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()