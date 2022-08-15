#!/usr/bin/env python3
import pandas as pd


DF_train = pd.read_csv("Train.csv", delimiter = ',')
DF_test = pd.read_csv('Test.csv', delimiter = ',')

DF_total = pd.concat([DF_train, DF_test])

#Удаление/добавление столбцов
DF_total = DF_total.drop(columns = 'Unnamed: 0', axis =1 )
DF_total = DF_total.drop(columns = 'period', axis =1 )

DF_total.info()

#One-hot
DF_one_hot = DF_total.copy()
DF_one_hot = pd.get_dummies(DF_one_hot)
DF_one_hot.tail()

DF_one_hot.to_csv('One_hot.csv',index=False)
