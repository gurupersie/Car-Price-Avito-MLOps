'''
Программа получения метрик.
Версия: 0.1.
'''

from typing import Dict
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import yaml


def get_metrics(y_train: pd.Series, y_test: pd.Series, y_pred: np.array,
                x_train: pd.DataFrame, x_test: pd.DataFrame, model: LGBMRegressor) -> Dict:
    '''
    Функция для получения метрик по предсказаниям модели.
    :param y_test:  истинные значения целевой переменной.
    :param y_pred:  предсказанные значения целевой переменной.
    :param x_test:  тестовая выборка признаков.
    :param model:  модель.
    :return: датафрейм с метриками.
    '''
    def r2_adjusted(y_test: pd.Series, y_pred: np.ndarray,
                    x_test: pd.DataFrame) -> float:
        """
        Коэффициент детерминации (множественная регрессия).
        :param y_test:  истинные значения целевой переменной.
        :param y_pred:  предсказанные значения целевой переменной.
        :param x_test:  тестовая выборка признаков.
        :return: значение коэффициента детерминации.
        """
        n_objects = len(y_test)
        n_features = x_test.shape[1]
        r2 = r2_score(y_test, y_pred)
        return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)

    def wape(y_test: pd.Series, y_pred: np.ndarray) -> float:
        """
        Weighted Absolute Percent Error.
        :param y_test: pd.Series - истинные значения целевой переменной.
        :param y_pred: np.array - предсказанные значения целевой переменной.
        :return: значение ошибки WAPE.
        """
        return np.sum(np.abs(y_pred - y_test)) / np.sum(y_test) * 100


    try:
        r2 = r2_adjusted(y_test, y_pred, x_test)
        wape = wape(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        dict_metrics = {
            'MAE': mae,
            'R2_Adjusted': r2,
            'WAPE': wape,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        dict_metrics.update(check_overfitting(x_train, y_train, x_test, y_test, model))

    except Exception as err:
        print(f'Ошибка в получении метрик {err}.')
    return dict_metrics


def check_overfitting(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                      y_test: pd.Series, model: LGBMRegressor) -> dict:
    '''
    Функция проверки на переобучение. Считает разницу
    в процентах MAE на тренировочной и тестовой выборках.
    :param x_train: обучающая выборка признаков.
    :param y_train: обучающая выборка целевой переменной.
    :param x_test: тестовая выборка признаков.
    :param y_test: тестовая выборка целевой переменной.
    :param model: экземляр класса обученной модели.
    :return: словарь со значением метрик для определения переобучения.-
    '''
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    mae_train = mean_absolute_error(y_train, train_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    percent_diff = np.round(abs((mae_test - mae_train) / mae_test * 100), 2)
    metric_dict = {
        'MAE train': mae_train,
        'MAE test': mae_test,
        'MAE diff, %': percent_diff
    }
    return metric_dict


def save_metrics(y_train: pd.Series, y_test: pd.Series, y_pred: np.array,
                 x_train: pd.DataFrame, x_test: pd.DataFrame, metrics_path: str,
                 model: LGBMRegressor) -> None:
    '''
    Сохранение метрик в json-файл по указанному пути.
    :param y_test: истинные значения целевой переменной.
    :param y_pred: предсказанные значения целевой переменной.
    :param x_test: тестовая выборка объект-признаков.
    :param model_path: путь для сохранения.
    :param model: модель.
    '''
    result_metrics = get_metrics(y_train, y_test, y_pred, x_train, x_test, model)

    with open(metrics_path, 'w', encoding='utf-8') as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> pd.DataFrame:
    '''
    Загрузка метрик из указанного пути.
    :param config_path: путь для загрузки.
    :return: датафрейм с метриками.
    '''
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['train']['metrics_path'], 'r', encoding='utf-8') as json_file:
        metrics = json.load(json_file)

    return metrics


def concat_metrics(dataframe: pd.DataFrame, y_true: np.array, y_pred: np.array,
                   X_test: np.array, model_name: str) -> pd.DataFrame:
    '''Функция для добавление метрик новых моделей в датасет с метриками. '''
    dataframe = pd.concat([
        dataframe,
        get_metrics_to_concat(y_true, y_pred, X_test, model_name=model_name)
    ])

    return dataframe


def get_metrics_to_concat(y_true: np.array, y_pred: np.array, X_test: np.array,
                model_name: str) -> pd.DataFrame:
    '''Функция для получения метрик по предсказаниям модели. '''
    def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray,
                    X_test: np.ndarray) -> float:
        """ Коэффициент детерминации (множественная регрессия). """
        n_objects = len(y_true)
        n_features = X_test.shape[1]
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)

    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Weighted Absolute Percent Error. """
        return np.sum(np.abs(y_pred - y_true)) / np.sum(y_true) * 100

    try:
        r2 = r2_adjusted(y_true, y_pred, X_test)
        wape = wape(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        df_metrics = pd.DataFrame({
            'Модель': [model_name],
            'MAE': [mae],
            'R2_Adjusted': [r2],
            'WAPE': [wape],
            'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))]
        })

    except Exception as err:
        df_metrics = pd.DataFrame({
            'Модель': [model_name],
            'MAE': [err],
            'R2_Adjusted': [err],
            'WAPE': [err],
            'RMSE': [err]
        })

    return df_metrics
