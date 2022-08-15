#!/usr/bin/env python3
import pandas as pd
from model_preparation import model, X_test

#Предсказание на тестовых данных
y_test=model.predict(X_test)

#Записть в столбец 'polution'
Submission = pd.DataFrame(y_test, columns = ["polution"])

#Перезапись в файл
Submission.to_csv('My_Submission.csv',index=False)
print(Submission.head())
