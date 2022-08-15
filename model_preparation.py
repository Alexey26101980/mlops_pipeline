#!/usr/bin/env python3
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge

DF_total = pd.read_csv('One_hot.csv')
DF_train = pd.read_csv('Train.csv', delimiter = ',')
DF_test = pd.read_csv('Test.csv', delimiter = ',')
Target = pd.read_csv('Target.csv', delimiter = ',')

#Посчет числа численных и категориальных колонок
cat_columns = []
num_columns = []

for column_name in DF_total.columns:
    if (DF_total[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]

#Разбивка данных обратно на Тренировочную и Тестовую
train = DF_total.iloc[0:DF_train.shape[0],:]
test = DF_total.iloc[DF_train.shape[0]:,:]

#Проверяем количество строк
DF_train.shape[0], train.shape[0], DF_test.shape[0], test.shape[0]

#Берем числовые данные
X_train = train[num_columns].values
X_test = test[num_columns].values

y_train = Target['polution'].values

#Разбиваем тренировочные данные на тренировочную и валидационную
X_train_, X_val, y_train_, y_val = train_test_split(X_train,y_train,test_size = 0.2, random_state = 42)
#Делаем предсказание
LR = LinearRegression(fit_intercept=True)

LR.fit(X_train_, y_train_)
#Результаты Кросс-валидации
scoring = {'R2': 'r2',
           '-MSE': 'neg_mean_squared_error',
           '-MAE': 'neg_mean_absolute_error',
           'Max': 'max_error'}


scores = cross_validate(LR, X_train_, y_train_,
                      scoring=scoring, cv=ShuffleSplit(n_splits=5, random_state=42) )

if __name__ == "__main__":
    print('Результаты Кросс-валидации')
    DF_cv_linreg = pd.DataFrame(scores)

    print('\n')
    print(DF_cv_linreg.mean()[2:])
    print('\n')

#Применение метрик
LR.fit(X_train_, y_train_)
y_predict=LR.predict(X_val)

if __name__ == "__main__":
    print('Ошибка на валидационных данных')
    print('MSE: %.1f' % mse(y_val,y_predict))
    print('RMSE: %.1f' % mse(y_val,y_predict,squared=False))
    print('R2 : %.4f' %  r2_score(y_val,y_predict))

#Регуларизация
#@title Регуларизация Ridge { run: "auto" }
#@markdown ### Константа Регуларизации
alpha=0.01 #@param {type:"slider", min:0.01, max:250, step:1}

model = Ridge(alpha=alpha,max_iter=10000 )

model.fit(X_train_, y_train_)

if __name__ == "__main__":
    print('Ошибка на тестовых данных')
    print('MSE: %.1f' % mse(y_val,y_predict))
    print('RMSE: %.1f' % mse(y_val,y_predict,squared=False))
    print('R2 : %.4f' %  r2_score(y_val,y_predict))
