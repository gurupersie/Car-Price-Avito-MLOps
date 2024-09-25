'''
Программа разбиения данных на train и test выборки.
Версия: 0.1
'''

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset_holdout(dataset: pd.DataFrame, stratify: pd.Series = None, **kwargs) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Разбиение данных на train/test и их сохранение в csv-файл.
    :param dataset: датасет.
    :param kwargs: параметры для train_test_split.
    :return: датасеты train и test.
    '''
    df_train, df_test = train_test_split(dataset,
                                         shuffle=True,
                                         stratify=stratify,
                                         test_size=kwargs['test_size'],
                                         random_state=kwargs['random_state'])
    return df_train, df_test


def get_train_test(data_train: pd.DataFrame, data_test: pd.DataFrame, target: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Получение тренировочных и тестовых выборок для обучения модели.
    :param data_train: тренировочный датасет.
    :param data_test: тестовый датасет.
    :param target: целевая переменная.
    :return: выборки x_train/x_test/y_train/y_test.
    '''
    x_train, x_test = (
        data_train.drop(target, axis=1),
        data_test.drop(target, axis=1)
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target]
    )
    return x_train, x_test, y_train, y_test
