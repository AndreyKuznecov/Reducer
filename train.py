"""
Данный файл предназначен для обучения модели и выгрузки ее параметров в отдельный файл
"""

import pandas as pd
import numpy as np
import sqlalchemy as sql
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

#выгрузка данных
try:
    engine = sql.create_engine('mysql://root:3456@127.0.0.1:3306/reducer_db')
    df = pd.read_sql('SELECT * FROM reducers', engine)
except Exception as e:
    raise Exception("Ошибка выгрузки данных") from e

df.drop(columns = ['id', 'i_f', 'J1'], inplace = True)
df['Fr1'].fillna(method = 'bfill', inplace = True)

#удаление выбросов
i = df["i"].quantile(0.95)
n2 = df["n2"].quantile(0.95)
P1 = df["P1"].quantile(0.95)
Fr2_high = df[df['m']>80]['Fr2'].quantile(0.975)
Fr2_low = df[df['m']>80]['Fr2'].quantile(0.025)
M2_high = df[df['m']>80]['M2'].quantile(0.975)
M2_low = df[df['m']>80]['M2'].quantile(0.025)
mask = ((df['i'] > i)
        | (df['n2'] > n2)
        | (df['P1'] > P1)
        | ((df['Fr2'] > Fr2_high) & (df['m']>80))
        | ((df['Fr2'] < Fr2_low) & (df['m']>80))
        | ((df['M2'] > M2_high) & (df['m']>80))
        | ((df['M2'] < M2_low)) & (df['m']>80))
data = df[~mask]

y = data['m']
X = data.drop("m", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#определение трансформеров
col_degree = ColumnTransformer([
    ('sqrt',FunctionTransformer(np.sqrt, feature_names_out = 'one-to-one'), slice(1, None)),
    ('poly',PolynomialFeatures(degree = 2),slice(1, None))], remainder = 'passthrough')

col_tran_new = ColumnTransformer([
    ('num',StandardScaler(),slice(None, -1)),
    ('cat',OneHotEncoder(categories = [X_train['type'].unique()]),[-1])], verbose_feature_names_out = False)

#создание пайплайна
steps_lasso=[('col_degree',col_degree), ('col_tran_new',col_tran_new), ('lasso', Lasso())]
pipe_lasso = Pipeline(steps=steps_lasso)

#поиск лучшей модели
col_degree__transformers = [[('sqrt', FunctionTransformer(feature_names_out='one-to-one', func=np.sqrt),
                        slice(1, None, None)),('poly', PolynomialFeatures(degree=i), slice(1, None, None))]for i in range(1,6)]\
                        +[[('poly', PolynomialFeatures(degree=i), slice(1, None, None))]for i in range(1,6)]

param_grid = {'lasso__alpha':list(np.logspace(-4, 2, 20)),
              'col_degree__transformers': col_degree__transformers}

grid_lasso = GridSearchCV(estimator = pipe_lasso, param_grid = param_grid, cv = 5, verbose=3)
grid_lasso.fit(X_train, y_train)

#сохранение лучшей модели
lasso_best = grid_lasso.best_estimator_
joblib.dump(lasso_best, r'models\lasso_model.pkl')