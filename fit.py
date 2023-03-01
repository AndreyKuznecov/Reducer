"""
Данный файл предназначен для предсказания значений массы на обученной модели
Входные данные извлекаются из файла data\input.csv, выходные записываются в data\output.csv
"""
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

try:
    input_path = os.path.join('data', 'input.csv')
    output_path = os.path.join('data', 'output.csv')
    #чтение входных данных
    if not os.path.isfile(input_path):
        raise FileNotFoundError('Входной файл не найден')
    data = pd.read_csv(input_path, sep=';')
    if data.shape[0] == 0 or data.shape[1] != 7:
        raise Exception('Файл пуст или неверной размерности')

    #загрузка модели и получение предсказаний
    model_path = os.path.join('models', 'lasso_model.pkl')
    model = joblib.load(model_path)
    y_pred = model.predict(data)

    #выгрузка результатов
    data['weight'] = y_pred
    data.to_csv(output_path, sep=';', index=False)
except Exception as e:
    print(e)