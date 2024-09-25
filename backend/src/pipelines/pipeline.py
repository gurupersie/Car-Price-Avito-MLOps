'''
Программа конвеера для обработки данных и тренировки модели.
Версия: 0.1.
'''

import os
import yaml
import joblib

from ..data.read_data import read_data
from ..train.train import train_model, find_best_params
from ..transform.transform import preprocessing_pipeline


def training_pipeline(config_path: str) -> None:
    '''
    Цикл: получение данных, предобработка, обучение модели.
    :param config_path: путь до конфигурационного файла.
    :return: None.
    '''
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preproc_config = config['preprocessing']
    train_config = config['train']

    train_data = read_data(preproc_config['raw_data_path'])
    data_train, data_test = preprocessing_pipeline(data=train_data,
                                                   save_csv=True,
                                                   flg_evaluate=False,
                                                   **preproc_config)

    study = find_best_params(df_train=data_train, df_test=data_test, **train_config)
    regressor = train_model(df_train=data_train,
                            df_test=data_test,
                            target=train_config['target_column'],
                            study=study,
                            metrics_path=train_config['metrics_path'])

    joblib.dump(regressor, os.path.join(train_config['models_path']))
    joblib.dump(study, os.path.join(train_config['study_path']))
