# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:41:37 2019

@author: jannett chabbeh
"""

import pymongo 
import math
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

client = pymongo.MongoClient("mongodb://localhost:27017/")
db= client['financial']
account=db['account']
cur= account.find({"_id": 244})
for x in cur:
    datex=x['date']
def getData():
   trans=db['transaction']
   myquery={ "account_id":244 }
   tab_dates,tab_bals =[],[]
   for x in trans.find(myquery):  
        #print("d: {} , a: {}, b: {}".format(x['date'],x['type']* x['amount'], x['balance']))
        d=x['date']-datex
        #print(int(d.days))
        tab_dates.append([d.days])
        tab_bals.append([x['balance']])
        
   return np.array(tab_dates), np.array(tab_bals)   

    
tab_dates,tab_bals=getData()
#auto_deprecated
model = LinearRegression()
print(model)

model.fit(tab_dates,tab_bals)
pred_bals = model.predict(tab_dates)
#print(pred_bals)
for yo, yp in zip(tab_bals[1:15,:], pred_bals[1:15]):
    print(yo,yp)

plt.scatter(tab_dates, tab_bals, color = "m", marker = "o", s = 10)
plt.plot(tab_dates, pred_bals, lw=1.5, color="red", label="predicted")
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(tab_bals,pred_bals))
r2 = r2_score(tab_bals,pred_bals)
print("Root Mean Squared Error :", rmse)
print("score of linear regression: ",r2)