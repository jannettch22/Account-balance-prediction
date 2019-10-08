# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:43:08 2019

@author: jannett chabbeh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:40:51 2019

@author: jannett chabbeh
"""
import pymongo 
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math


client = pymongo.MongoClient("mongodb://localhost:27017/")
db= client['financial']
account=db['account']
cur= account.find({"account_id": 200})
for x in cur:
    datex=x['date']
trans=db['transaction']
myquery={ "account_id":200, 'type':1  }
series =[]
dates=[]
for x in trans.find(myquery):  
    d=x['date']-datex
    series.append( x['amount']*x['type'])
    dates.append(d.days)

X = series

size = int(len(X) * 0.66)

train, test = X[0:size], X[size:len(X)]
dates_test=dates[size:len(dates)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f, average= %f ' % (yhat, obs, obs-yhat))

#print(score)
# plot
pyplot.plot(dates_test,test, lw=1.5, color="blue", label="excepted")
pyplot.plot(dates_test,predictions, color='red')
pyplot.show()
mse = mean_squared_error(test, predictions)
print("Mean Squared Error:",mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)