'''
Программа получения предсказаний обученной модели.
Версия: 0.1.
'''

import os
import pandas as pd
import joblib
import yaml
from ..data.read_data import read_data
from ..transform.transform import preprocessing_pipeline


def evaluation_pipeline(config_path: str, data: pd.DataFrame = None, data_path: str = None) -> list:
    '''
    Предобработка данных и получение предсказаний модели.
    :param config_path: путь к конфигурационному файлу.
    :param data: датафрейм.
    :param data_path: путь к данным.
    :return: список с предсказаниями.
    '''

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preproc_config = config['preprocessing']
    train_config = config['train']

    if data_path:
        data = read_data(data_path)

    data = preprocessing_pipeline(data=data, **preproc_config)

    model = joblib.load(os.path.join(train_config['models_path']))
    prediction = model.predict(data).round(0).tolist()
    return prediction