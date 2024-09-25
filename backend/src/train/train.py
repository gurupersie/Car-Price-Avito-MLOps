'''
Программа обучения модели на тренировочных данных.
Версия: 0.1
'''

import pandas as pd
import numpy as np

import optuna
from optuna_integration import LightGBMPruningCallback
from optuna import Study

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor

from ..data.split_data import get_train_test
from ..train.metrics import save_metrics


def objective(trial, x: pd.DataFrame, y: pd.Series,
              n_folds: int = 5, random_state: int = 1618) -> np.array:
    '''
    Функция для подбора гиперпараметров через фреймворк Optuna для LGBMRegressor.
    :param trial: количество испытаний.
    :param X: матрица объект-признаков.
    :param y: целевая переменная.
    :param n_folds количество фолдов.
    :param random_state: фиксирование случайности.
    :return: среднее значение MAE по фолдам.
    '''
    params = {
        'n_estimators':
            trial.suggest_categorical('n_estimators', [1000]),
        'learning_rate':
            trial.suggest_float('learning_rate', 1e-4, 0.2, log=True),
        'max_depth':
            trial.suggest_int('max_depth', 3, 13),
        'lambda_l1':
            trial.suggest_float('lambda_l1', 0, 100),
        'lambda_l2':
            trial.suggest_float('lambda_l2', 0, 100),
        'feature_fraction':
            trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_freq':
            trial.suggest_int('bagging_freq', 0, 15),
        'bagging_fraction':
            trial.suggest_float('bagging_fraction', 0.05, 0.9),
        'extra_trees':
            trial.suggest_categorical('extra_trees', [True]),
        'extra_seed':
            trial.suggest_categorical('extra_seed', [random_state]),
        'colsample_bytree':
            trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'num_leaves':
            trial.suggest_int('num_leaves', 2, 2 ** 13),
        'min_split_gain':
            trial.suggest_int('min_split_gain', 0, 20),
        'verbose':
            trial.suggest_categorical('verbose', [-1]),
        'early_stopping_rounds':
            trial.suggest_categorical('early_stopping_rounds', [50]),
        'random_state':
            trial.suggest_categorical('random_state', [random_state]),
    }

    cv_opt = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_predicts = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv_opt.split(x, y)):
        x_train, x_val = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        pruning_callback = LightGBMPruningCallback(trial, metric='l1')

        model = LGBMRegressor(**params)
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_val, y_val)],
                  eval_metric='mae',
                  callbacks=[pruning_callback])

        y_pred = model.predict(x_val)

        cv_predicts[idx] = mean_absolute_error(y_val, y_pred)

    return np.mean(cv_predicts)


def find_best_params(df_train: pd.DataFrame, df_test: pd.DataFrame, **kwargs) -> Study:
    '''
    Поиск лучших параметров для модели.
    :param df_train: тренировочный датасет.
    :param df_test: тестовый датасет.
    :param kwargs: гиперпараметры из конфигурационного файла.
    :return: [LGBMRegressor tuning, Study]
    '''
    x_train, _, y_train, _ = get_train_test(df_train,
                                            df_test,
                                            target=kwargs['target_column'])
    sampler = optuna.samplers.TPESampler(seed=kwargs['random_state'])
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize',
                                sampler=sampler,
                                pruner=pruner,
                                study_name='LGBM')
    function = lambda trial: objective(
        trial, x_train, y_train, n_folds=kwargs['n_folds'], random_state=kwargs['random_state']
    )
    study.optimize(function, n_trials=kwargs['n_trials'], show_progress_bar=True, n_jobs=-1)
    return study


def train_model(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str,
                study: Study, metrics_path: str) -> LGBMRegressor:
    '''
    Обучение модели на тренировочных данных и сохранение метрик по указанному пути.
    :param df_train: тренировочный датасет.
    :param df_test: тестовый датасет.
    :param target: целевая переменная.
    :param study: объект optuna, предварительно обученный
                  на тестовой выборке для подбора лучших гиперпараметров.
    :param metrics_path: путь для сохранения метрик.
    :return: обученная модель LGBMRegressor.
    '''
    x_train, x_test, y_train, y_test = get_train_test(df_train, df_test, target)
    indexes = np.random.choice(x_train.index.values, size=1500)
    x_train_, x_val, y_train_, y_val = x_train[~x_train.index.isin(indexes)], \
                                       x_train[x_train.index.isin(indexes)], \
                                       y_train[~x_train.index.isin(indexes)], \
                                       y_train[x_train.index.isin(indexes)]

    model = LGBMRegressor(**study.best_params)
    model.fit(x_train_, y_train_, eval_set=[(x_val, y_val)], eval_metric='mae')
    save_metrics(y_train=y_train,
                 y_test=y_test,
                 y_pred=model.predict(x_test),
                 x_train=x_train,
                 x_test=x_test,
                 metrics_path=metrics_path,
                 model=model)
    return model
