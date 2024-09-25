'''
Программа модели для предсказания стоимости автомобиля.
Версия: 0.1
'''

import warnings
import pandas as pd
import optuna

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import training_pipeline
from src.evaluate.evaluate import evaluation_pipeline
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATH = '../config/parameters.yml'
app = FastAPI()


class CarPrice(BaseModel):
    '''
    Признаки для получения результатов модели.
    '''
    Рейтинг: float
    Год_выпуска: int
    Поколение: str
    Пробег_км: int
    История_пробега_кол_во_записей: int
    Владельцев_по_ПТС: int
    Состояние: str
    Модификация: str
    Объём_двигателя_л: float
    Тип_двигателя: str
    Коробка_передач: str
    Привод: str
    Комплектация: str
    Тип_кузова: str
    Цвет: str
    Руль: str
    Управление_климатом: str
    ПТС: str
    Обмен: int
    Бренд_авто: str
    Модель_авто: str
    Город: str
    Федеральный_округ: str
    Выпуск_кол_во_лет: int
    Мощность_двигателя_лс: int
    Много_владельцев: int
    Срок_эксплуатации: int
    Степень_износа: str


@app.get('/hello')
def welcome_message():
    '''
    Приветственное сообщение.
    '''
    return {'message': 'Добрейшего дня!'}


@app.post('/train')
def training() -> dict:
    '''
    Обучение модели и логирование метрик.
    :return: dict с метриками.
    '''
    training_pipeline(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)
    return {'metrics': metrics}


@app.post('/prediction_from_file')
def prediction(file: UploadFile = File(...)):
    '''
    Предсказание модели по данным из файла.
    :param file: поле для загрузки файла с данными.
    :return: dict с результатом предсказаний.
    '''
    result = evaluation_pipeline(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), 'Неверный тип данных, result должен быть list.'
    return {'prediction': result[:5]}


@app.post('/prediction_by_input')
def input_prediction(car: CarPrice):
    '''
    Предсказание модели по введенным вручную данным.
    :param car: класс для проверки типов данных
    :return: dict с ценой автомобиля.
    '''
    features = [
        [
            car.Рейтинг,
            car.Год_выпуска,
            car.Поколение,
            car.Пробег_км,
            car.История_пробега_кол_во_записей,
            car.Владельцев_по_ПТС,
            car.Состояние,
            car.Модификация,
            car.Объём_двигателя_л,
            car.Тип_двигателя,
            car.Коробка_передач,
            car.Привод,
            car.Комплектация,
            car.Тип_кузова,
            car.Цвет,
            car.Руль,
            car.Управление_климатом,
            car.ПТС,
            car.Обмен,
            car.Бренд_авто,
            car.Модель_авто,
            car.Город,
            car.Федеральный_округ,
            car.Выпуск_кол_во_лет,
            car.Мощность_двигателя_лс,
            car.Много_владельцев,
            car.Срок_эксплуатации,
            car.Степень_износа
        ]
    ]

    columns = [
        'Рейтинг',
        'Год_выпуска',
        'Поколение',
        'Пробег_км',
        'История_пробега_кол_во_записей',
        'Владельцев_по_ПТС',
        'Состояние',
        'Модификация',
        'Объём_двигателя_л',
        'Тип_двигателя',
        'Коробка_передач',
        'Привод',
        'Комплектация',
        'Тип_кузова',
        'Цвет',
        'Руль',
        'Управление_климатом',
        'ПТС',
        'Обмен',
        'Бренд_авто',
        'Модель_авто',
        'Город',
        'Федеральный_округ',
        'Выпуск_кол_во_лет',
        'Мощность_двигателя_лс',
        'Много_владельцев',
        'Срок_эксплуатации',
        'Степень_износа'
    ]

    data = pd.DataFrame(data=features, columns=columns)
    prediction = evaluation_pipeline(config_path=CONFIG_PATH, data=data)
    return f'Цена автомобиля: {prediction[0]}.'


if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)
