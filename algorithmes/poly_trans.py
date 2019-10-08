# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:20:24 2019

@author: jannett chabbeh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:50:41 2019

@author: jannett chabbeh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:50:41 2019

@author: jannett chabbeh
"""

import pymongo 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
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
        tab_bals.append([x['amount']])
        
   return np.array(tab_dates), np.array(tab_bals) 

    
tab_dates,tab_bals=getData()

polynomial_features= PolynomialFeatures(degree=5)
x_poly = polynomial_features.fit_transform(tab_dates)

model = LinearRegression()
model.fit(x_poly, tab_bals)
y_poly_pred = model.predict(x_poly)
size= int(len(tab_bals) * 0.66)
test_exce= tab_bals[size:len(tab_bals)]
test_pred =  y_poly_pred[size:len(tab_bals)]
dtest=  tab_dates[size:len(tab_dates)] 

plt.plot(dtest, test_exce,lw=1.5, color="m", label="excepted")
plt.plot(dtest, test_pred, lw=1.5, color="red", label="predicted")
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(tab_bals,y_poly_pred))
r2 = r2_score(tab_bals,y_poly_pred)
print("Root Mean Squared Error :", rmse)
print("score of linear regression: ",r2)