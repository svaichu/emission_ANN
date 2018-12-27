#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 20:44:04 2018

@author: vaishnavahari
"""

import csv
#datafile = open('a.csv', 'r')
#datareader = csv.reader(datafile, delimiter=';')
#data = []
#for row in datareader:
#    data.append(row)
#data = list(csv.reader(open('Emission Data - data_prepared.csv')))

#result = [list( map(float,i) ) for i in data]

import pandas as pd
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('Emission Data - data_prepared.csv')

X = df[['P_inj_bar','imep_bar','diesel_percentage','cookingoil_percentage','EGR_percentage','AFR']]
y = df[[' CO2_percentage','NOx_ppm','Soot_[mg/m^3]']]

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_x.fit(X)
X = scaler_x.transform(X)
scaler_y = StandardScaler()
scaler_y.fit(y)
y = scaler_y.transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

reg = MLPRegressor(hidden_layer_sizes=(5,),
                                     activation='relu',
                                     solver='adam',
                                     learning_rate='adaptive',
                                     max_iter=1000,
                                     learning_rate_init=0.01,
                                     alpha=0.01)
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test)
print(reg.score(X_test, y_test))
y_predict = scaler_y.inverse_transform(y_predict)

#from ann_visualizer.visualize import ann_viz;
#Build your model here
#ann_viz(reg)

print(reg)