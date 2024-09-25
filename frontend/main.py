'''
Программа Frontend.
Версия: 0.1.
'''

import os
import yaml
import streamlit as st
from src.plotting.charts import plot_bars, corr_matrix, plot_countplot, plot_median_barplot
from src.data.get_data import read_data, load_data
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = '../config/parameters.yml'


def main_page():
    """
    Главная страница проекта с описанием.
    """
    st.image('https://cdnstatic.rg.ru/crop560x374/uploads/images/2022/08/10/3p_avto_1000_f0b.jpg')
    st.markdown('# Описание проекта')
    st.title('MLOps project: Предсказание цены автомобиля.')
    st.write(
        '''
        Данные получены при помощи парсинга сайта avito.ru. 
        Необходимо создать модель, способную по характеристикам автомобиля
        и дополнительным параметрам предсказать его цену. 
        '''
    )
    st.markdown(
        '''
        ### Данные по автомобилям:

        Целевая переменная - Цена.
        
        - **Рейтинг** - рейтинг автомобиля на сайте.
        - **Год выпуска** - год производства автомобиля.
        - **Поколение** - поколение автомобиля.
        - **Пробег, км** - сколько проехал автомобиль в километрах.
        - **История пробега, кол-во записей** - количество записей об истории пробега в автотеке. 
        - **Владельцев по ПТС** - количество владельцев, записанных в ПТС.
        - **Состояние** - общее состояние автомобиля.
        - **Модификация** - модификация автомобиля.
        - **Объём двигателя, л** - объём двигателя в литрах.
        - **Тип двигателя** - тип двигателя.
        - **Коробка передач** - тип коробки передач.
        - **Привод** - тип привода.
        - **Комплектация** - комплектация автомобиля
        - **Тип кузова** - тип кузова автомобиля.
        - **Цвет** - цвет автомобиля.
        - **Руль** - расположение руля.
        - **Управление климатом** - тип системы управления климатом
        - **ПТС** - вид ПТС.
        - **Обмен** - рассматривается ли обмен автомобиля на другой, или альтернативные варианты.
        - **Бренд авто** - наименование бренда авто, производитель.
        - **Модель авто** - наименование модели авто.
        - **Город** - город, в котором продаётся авто.
        - **Регион** - регион, в котором продаётся авто.
        - **Выпуск, кол-во лет** - количество лет, в течение которого шёл выпуск автомобиля
        - **Мощность двигателя, лс** - мощность двигателя автомобиля, в лошадиных силах.
        '''
    )


def exploratory():
    '''
    Разведочный анализ данных.
    '''
    st.markdown('# Exploratory data analysis.')

    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data = read_data(data_path=config['preprocessing']['train_path_proc'])
    st.write(data[:5])

    brand_counts = st.sidebar.toggle('Количество объектов в каждом бренде авто.')
    correlations = st.sidebar.toggle('Корреляция признаков.')
    median_price = st.sidebar.toggle('Медианная цена продажи по брендам.')
    transmission = st.sidebar.toggle('Типы коробок передач в разрезе брендов.')

    if brand_counts:
        st.pyplot(plot_countplot(df=data, feature_name='Бренд_авто'))
    if correlations:
        st.pyplot(corr_matrix(data=data))
    if median_price:
        st.pyplot(plot_median_barplot(df=data, group_feature='Бренд_авто', target_feature='Цена'))
    if transmission:
        st.pyplot(plot_bars(df=data, target='Бренд_авто', feature='Коробка_передач'))


def training():
    '''
    Обучение модели.
    '''
    st.markdown('# Обучение модели LGBMRegressor.')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['train']

    if st.button('Обучить модель.'):
        start_training(config=config, endpoint=endpoint)


def prediction_by_input():
    '''
    Получение предсказаний модели по введенным вручную данным.
    '''
    st.markdown('# Получение предсказаний модели по введенным вручную данным.')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_by_input']
    unique_data_path = config['preprocessing']['unique_values_path']
    cars_data_path = config['preprocessing']['cars_dict_path']

    if os.path.exists(config['train']['models_path']):
        evaluate_input(unique_data_path=unique_data_path,
                       cars_data_path=cars_data_path,
                       endpoint=endpoint)
    else:
        st.error('Сначала обучите модель.')


def prediction_from_file():
    '''
    Получение предсказаний модели из файла с данными.
    '''
    st.markdown('# Получение предсказаний модели из файла с данными.')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_from_file']
    uploaded_file = st.file_uploader('Выберите файл', type=['csv', 'xlsx'], accept_multiple_files=False)
    if uploaded_file:
        data_csv_df, files = load_data(uploaded_data=uploaded_file, data_type='test')
        if os.path.exists(config['train']['models_path']):
            evaluate_from_file(data=data_csv_df, endpoint=endpoint, files=files)
        else:
            st.error('Сначала обучите модель.')


def main():
    '''
    Окончательная сборка пайплайна в одном блоке.
    '''
    page_names_to_funcs = {
        'Описание проекта': main_page,
        'Разведочный анализ данных': exploratory,
        'Обучение модели': training,
        'Предсказание по введённым данным': prediction_by_input,
        'Предсказание по данным из файла': prediction_from_file,
    }
    selected_category = st.sidebar.radio('Выберите раздел:', page_names_to_funcs.keys())
    page_names_to_funcs[selected_category]()


if __name__ == '__main__':
    main()
