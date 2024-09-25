'''
Программа отрисовки слайдера и кнопок для ввода данных с дальнейшим
получением предсказания на основании введенных значений.
Версия: 0.1.
'''

import json
from io import BytesIO
import pandas as pd
import numpy as np
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, cars_data_path: str, endpoint: object) -> None:
    '''
    Получение входных данных путём ввода их значений в UI и вывод результата.
    :param unique_data_path: путь до уникальных значений.
    :param endpoint: endpoint.
    :param rating_dict_path: путь до словаря с рейтингом.
    '''
    assert isinstance(unique_data_path, str), 'Проверьте unique_data_path, должен быть str.'
    assert isinstance(endpoint, object), 'Проверьте endpoint, должен быть object.'
    with open(unique_data_path, 'r', encoding='utf-8') as file:
        unique_data = json.load(file)
    with open(cars_data_path, 'r', encoding='utf-8') as file:
        cars_data = json.load(file)

    car_brand = st.sidebar.selectbox('Бренд_авто', (list(cars_data['Бренды'].keys())))
    car_model = st.sidebar.selectbox('Модель_авто', (list(cars_data['Бренды'][car_brand])))
    modification = st.sidebar.selectbox('Модификация', (list(cars_data['Модели'][car_model])))
    generation = st.sidebar.selectbox('Поколение', (list(cars_data['Поколения'][modification])))
    rating = cars_data['Рейтинги'][generation]
    if np.isnan(rating):
        rating = 4.5
    year = st.sidebar.selectbox('Год_выпуска', (unique_data['Год_выпуска']))
    mileage = st.sidebar.number_input('Пробег_км',
                                      min_value=min(unique_data['Пробег_км']),
                                      max_value=max(unique_data['Пробег_км']))
    history = st.sidebar.selectbox('История_пробега_кол_во_записей',
                                   (unique_data['История_пробега_кол_во_записей']))
    owners = st.sidebar.selectbox('Владельцев_по_ПТС',
                                  (unique_data['Владельцев_по_ПТС']))
    condition = st.sidebar.selectbox('Состояние', (unique_data['Состояние']))
    engine_volume = st.sidebar.slider('Объём_двигателя_л',
                               min_value=min(unique_data['Объём_двигателя_л']),
                               max_value=max(unique_data['Объём_двигателя_л']))
    engine_type = st.sidebar.selectbox('Тип_двигателя', (unique_data['Тип_двигателя']))
    transmission = st.sidebar.selectbox('Коробка_передач',
                                        (unique_data['Коробка_передач']))
    drive_unit = st.sidebar.selectbox('Привод', (unique_data['Привод']))
    equipment = st.sidebar.selectbox('Комплектация', (unique_data['Комплектация']))
    body_type = st.sidebar.selectbox('Тип_кузова', (unique_data['Тип_кузова']))
    color = st.sidebar.selectbox('Цвет', (unique_data['Цвет']))
    rudder = st.sidebar.selectbox('Руль', (unique_data['Руль']))
    climate_control = st.sidebar.selectbox('Управление_климатом',
                                           (unique_data['Управление_климатом']))
    vehicle_passport = st.sidebar.selectbox('ПТС', (unique_data['ПТС']))
    exchange = st.sidebar.selectbox('Обмен', (unique_data['Обмен']))
    city = st.sidebar.selectbox('Город', (unique_data['Город']))
    federal_district = st.sidebar.selectbox('Федеральный_округ',
                                            (unique_data['Федеральный_округ']))
    production_duration = st.sidebar.selectbox('Выпуск_кол_во_лет',
                                               (unique_data['Выпуск_кол_во_лет']))
    engine_power = st.sidebar.slider('Мощность_двигателя_лс',
                               min_value=min(unique_data['Мощность_двигателя_лс']),
                               max_value=max(unique_data['Мощность_двигателя_лс']))
    many_owners = st.sidebar.selectbox('Много_владельцев',
                                       (unique_data['Много_владельцев']))
    lifetime = st.sidebar.slider('Срок_эксплуатации',
                               min_value=min(unique_data['Срок_эксплуатации']),
                               max_value=max(unique_data['Срок_эксплуатации']))
    degree_of_wear = st.sidebar.selectbox('Степень_износа',
                                          (unique_data['Степень_износа']))

    data_dict = {
        'Рейтинг': rating,
        'Год_выпуска': year,
        'Поколение': generation,
        'Пробег_км': mileage,
        'История_пробега_кол_во_записей': history,
        'Владельцев_по_ПТС': owners,
        'Состояние': condition,
        'Модификация': modification,
        'Объём_двигателя_л': engine_volume,
        'Тип_двигателя': engine_type,
        'Коробка_передач': transmission,
        'Привод': drive_unit,
        'Комплектация': equipment,
        'Тип_кузова': body_type,
        'Цвет': color,
        'Руль': rudder,
        'Управление_климатом': climate_control,
        'ПТС': vehicle_passport,
        'Обмен': exchange,
        'Бренд_авто': car_brand,
        'Модель_авто': car_model,
        'Город': city,
        'Федеральный_округ': federal_district,
        'Выпуск_кол_во_лет': production_duration,
        'Мощность_двигателя_лс': engine_power,
        'Много_владельцев': many_owners,
        'Срок_эксплуатации': lifetime,
        'Степень_износа': degree_of_wear
    }

    st.write(
        f'''
        ### Данные автомобиля: \n
        1. Бренд_авто: {data_dict['Бренд_авто']}, 
        2. Модель_авто: {data_dict['Модель_авто']}, 
        3. Модификация: {data_dict['Модификация']}, 
        4. Поколение: {data_dict['Поколение']},  
        5. Год_выпуска: {data_dict['Год_выпуска']}
        6. Пробег_км: {data_dict['Пробег_км']}, 
        7. Степень_износа: {data_dict['Степень_износа']}, 
        8. Привод: {data_dict['Привод']}, 
        9. Тип_кузова: {data_dict['Тип_кузова']}, 
        10. Объём_двигателя_л: {data_dict['Объём_двигателя_л']}, 
        11. Коробка_передач: {data_dict['Коробка_передач']},
        12. Тип_двигателя: {data_dict['Тип_двигателя']},
        13. Мощность_двигателя_лс: {data_dict['Мощность_двигателя_лс']}, 
        14. Срок_эксплуатации: {data_dict['Срок_эксплуатации']},
        15. Рейтинг авто на Авито: {data_dict['Рейтинг']}, 
        16. Федеральный_округ: {data_dict['Федеральный_округ']}, 
        17. Владельцев_по_ПТС: {data_dict['Владельцев_по_ПТС']}, 
        18. Город: {data_dict['Город']}, 
        19. Комплектация: {data_dict['Комплектация']}, 
        20. Цвет: {data_dict['Цвет']},
        21. Много_владельцев: {data_dict['Много_владельцев']},
        22. Состояние: {data_dict['Состояние']},
        23. Обмен: {data_dict['Обмен']},
        24. ПТС: {data_dict['ПТС']}, 
        25. История_пробега_кол_во_записей: {data_dict['История_пробега_кол_во_записей']},
        26. Руль: {data_dict['Руль']},
        27. Выпуск_кол_во_лет: {data_dict['Выпуск_кол_во_лет']},
        28. Управление_климатом: {data_dict['Управление_климатом']},
        '''
    )

    button_predict = st.button('Предсказать')
    if button_predict:
        result = requests.post(endpoint, timeout=8000, json=data_dict)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f'## {output}')
        st.success('Успешно!')


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO) -> None:
    '''
    Получение входных данных через загрузку файла -> вывод результата в виде таблицы.
    :param data: датафрейм.
    :param endpoint: endpoint.
    :param files: битовое представление.
    '''
    assert isinstance(data, pd.DataFrame), 'Проверьте data, должен быть pd.DataFrame.'
    assert isinstance(endpoint, object), 'Проверьте endpoint, должен быть object.'
    button_predict = st.button('Предсказать')
    if button_predict:
        data_ = data[:5]
        output = requests.post(endpoint, timeout=8000, files=files)
        data_['Предсказанная_цена'] = output.json()['prediction']
        st.write(data_[:5])
