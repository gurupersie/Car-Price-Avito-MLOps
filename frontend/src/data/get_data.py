'''
Программа получения данных по указанному пути и чтения данных.
Версия: 0.1.
'''

from io import BytesIO
import io
from typing import Dict, Tuple, IO
import streamlit as st
import pandas as pd


def read_data(data_path: str) -> pd.DataFrame:
    '''
    Получение данных из файла по указанному пути.
    :param data_path: путь до файла.
    :return: датафрейм.
    '''
    assert isinstance(data_path, str), 'Проверьте путь, должен быть str.'
    return pd.read_csv(data_path)


def load_data(uploaded_data: IO, data_type: str = 'train'
              ) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    '''
    Получение данных и преобразование их в формат байтов тип Bytes IO для обработки в streamlit
    :param data_path: путь до данных.
    :param data_type: тип данных: тестовые или тренировочные (train/test).
    :return: датафрейм, датафрейм в формате BytesIO
    '''
    assert data_type in ['train', 'test'], 'Проверьте data_type, должен быть train/test.'
    data = pd.read_csv(uploaded_data)
    st.write('Данные загружены.')
    st.write(data[:5])

    data_bytes_object = io.BytesIO()
    data.to_csv(data_bytes_object, index=False)
    data_bytes_object.seek(0)

    files = {
        'file': (f'{data_type}_data.csv', data_bytes_object, 'multipart/form-data')
    }
    return data, files