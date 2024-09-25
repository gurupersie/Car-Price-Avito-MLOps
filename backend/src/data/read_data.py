'''
Программа чтения данных из файла csv.
Версия: 0.1
'''

import pandas as pd


def read_data(data_path: str) -> pd.DataFrame:
    '''
    Получение данных из заданого файла по пути data_path.
    :param data_path: путь до файла.
    :return: датафрейм.
    '''
    return pd.read_csv(data_path)
