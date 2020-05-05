import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

####importing the data set ####
#the data set is basically how mani  dollars are spent on adv in TV RADIO NEWSPAPERS AND THE SALES is the revenu generated in total

advert=pd.read_csv('Advertising.csv')

print(advert.head())
print(advert.info())

#### removing the index column ####

print(advert.columns)

advert.drop(['Unnamed: 0'],axis=1,inplace=True)

print(advert.head())

#### Exploratory Data Analysis####

import seaborn as sns

sns.distplot(advert.sales)
sns.distplot(advert.newspaper)
sns.distplot(advert.radio)
sns.distplot(advert.TV)

#### exploring relationship between predictors and response####

sns.pairplot(advert,x_vars=['TV','radio','newspaper'],y_vars='sales',height=7,aspect=0.7,kind='reg')

#we find thet the feaature tv is the best one to work with

print(advert.TV.corr(advert.sales))

print(advert.corr())

sns.heatmap(advert.corr(),annot=True)

#### creating the linear regression model####

x=advert[['TV']]
print(x.head())
print(type(x))
print(x.shape)

y=advert.sales

print(type(y))
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_train.shape)

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()

linreg.fit(x_train,y_train)


####interpreting model coef####

print(linreg.intercept_)
print(linreg.coef_)

#### making prediction ####

y_pred=linreg.predict(x_test)
print(y_pred[:5])

#### model eval metrics ####

true=[100,50,30,20]
pred=[90,50,50,30]

#print((10+0+20+10)/4)

from sklearn import metrics
#mean absolute error
print(metrics.mean_absolute_error(true,pred)) 

#mean squared error
print(metrics.mean_squared_error(true,pred))

#root mean squre error

print(np.sqrt(metrics.mean_squared_error(true,pred)))


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))







