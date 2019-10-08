# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:53:19 2019

@author: jannett chabbeh
"""

import pymongo 
import math
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVR 

from sklearn.metrics import mean_squared_error, r2_score

client = pymongo.MongoClient("mongodb://localhost:27017/")
db= client['financial']
account=db['account']
cur= account.find({"account_id": 200})
for x in cur:
    datex=x['date']
def getData():
   trans=db['transaction']
   myquery={ "account_id":200, 'type':1 }
   tab_dates,tab_bals =[],[]
   for x in trans.find(myquery):  
        #print("d: {} , a: {}, b: {}".format(x['date'],x['type']* x['amount'], x['balance']))
        d=x['date']-datex
        #print(int(d.days))
        tab_dates.append([d.days])
        tab_bals.append([x['amount']*x['type']])
        
   return np.array(tab_dates), np.array(tab_bals) 
tab_dates,tab_bals=getData()
size = int(len(tab_bals) * 0.66)

test =  tab_bals[size:len(tab_bals)]
dtest=  tab_dates[size:len(tab_dates)]   



#auto_deprecated ou scale
model = SVR(C=100000, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
  gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
  tol=0.001, verbose=False)
print(model)

model.fit(tab_dates,tab_bals)
pred_bals = model.predict(dtest)
#print(pred_bals)
n=model.predict([[800]])
print("la prediction: ",n)

#plt.scatter(tab_dates, tab_bals, color = "m", marker = "o", s = 10)
plt.plot(dtest, test,  lw=1.5, color="blue", label="excepted")
plt.plot(dtest, pred_bals, lw=1.5, color="red", label="predicted")
plt.legend()
plt.show()

score=model.score(dtest,test)
print("score of SVR: ",score)

mse =mean_squared_error(dtest, pred_bals)
print("Mean Squared Error:",mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)