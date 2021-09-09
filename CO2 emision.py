# Simple Linear Regression and Multiple Linear Regression


import pandas
import numpy as np
import matplotlib.pyplot as plt


''' Reading data from CSV'''                                                          #readinig


data=pandas.read_csv(r'C:\Users\tevit\Documents\V S C\ML\CO2 emision\Cars Data set.csv')
#print(data.shape)
print(data.columns)


'''data Visualization'''                                                #understanding the data

'''
scatter plots only works for 2dimentional that means we can only draw 
between 2 attributes
we can check the model prediction using scatter plots only in the simple linear regression
we can not use in multiple linear regression
'''
'''
x=data['Fuel Consumption Hwy (L/100 km)']
y=data['CO2 Emissions(g/km)']
plt.scatter(x,y)                        #check with different different fields to understand the data
plt.xlabel('fuel consumption')          #which is related and which is not so that 
plt.ylabel('co2 emission')              #we can apply the preprocessing
plt.title('scatter plot')
plt.show()
'''


'''data Predictions'''

#x=data.iloc[:,:-1].values
#y=data.iloc[:,-1].values


'''data spliting'''

#from sklearn.model_selection import train_test_split

#x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.20)

#another way of splitting

msk=np.random.rand(len(data))<0.8
train=data[msk]
test=data[~msk]


'''building model'''


from sklearn import linear_model

mod=linear_model.LinearRegression()

train_X=np.asanyarray(train[['Engine Size(L)','Fuel Consumption Comb (L/100 km)','Cylinders']])
train_Y=np.asanyarray(train[['CO2 Emissions(g/km)']])

mod.fit(train_X,train_Y)

print(mod.coef_,mod.intercept_)   #coef_ gives 2d array intercept_ givees 1d

'''testing'''

test_X=np.asanyarray(test[['Engine Size(L)','Fuel Consumption Comb (L/100 km)','Cylinders']])
test_Y=np.asanyarray(test[['CO2 Emissions(g/km)']])

predict_Y=mod.predict(test_X)

   
plt.scatter(train_X,train_Y,color='black')
plt.scatter(train_X,mod.coef_[0][0]*train_X+mod.intercept_,color='red')
plt.xlabel('engine size ')
plt.ylabel('co2 emission')
plt.show()


#testing

'''mod.coef_[0][0]*test_X+mod.intercept_'''
'''
plt.scatter(test_X,predict_Y,color='blue')
plt.scatter(test_X,test_Y,color='Yellow')
plt.show()
'''
#checking performance

from sklearn.metrics import r2_score    

#max r2_score will be 1.0 then the accurecy will be 100%

print("mean diff :",np.mean(np.abs(test_Y,predict_Y)))
print("MSE :",np.mean((test_Y-predict_Y)**2))
print("r2_score :",r2_score(predict_Y,test_Y))
