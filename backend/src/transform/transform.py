'''
Программа предобработки данных.
Версия: 0.1.
'''

import warnings
import json
import re
from typing import Any, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from ..data.split_data import split_dataset_holdout

warnings.filterwarnings('ignore')


def transform_types(data: pd.DataFrame, types_dict: dict, err: str = 'raise',
                    object_to_category: bool = False) -> pd.DataFrame:
    '''
    Функция преобразования типов признаков в датасете.
    :param data: исходный датафрейм.
    :param types_dict: словарь с названиями признаков и их типами.
    :param err: параметр для вывода или игнорирования ошибок.
    :param object_to_category: трансформировать ли 'object' в 'category'.
    :return: трансформированный датафрейм.
    '''
    for col_name in types_dict.keys():
        if col_name in data.columns:
            data[col_name] = data[col_name].astype(types_dict[col_name], errors=err)
    if object_to_category:
        data[data.select_dtypes('object').columns] = data.select_dtypes('object').astype('category')
    return data


def modification_fill(mod_str: str) -> str:
    '''
    Функция для возврата подстроки из заданной строки.
    :param mod_str: значение признака.
    :return: подстрока с названием модификации.
    '''
    assert isinstance(mod_str, str), 'Проблема с данными на входе, mod_str должен быть str.'
    first_split = mod_str.split(',')[0]
    second_split = first_split.split()
    if '.' in first_split:
        return ' '.join(x for x in second_split[-2:])
    return second_split[-1]

def apply_function(function: Any, data: pd.DataFrame,
                   result_column: str, applying_column: str) -> pd.DataFrame:
    '''
    Функция для применения к признаку функции для его
    преобразования или заполнения пропусков.
    :param function: применяемая функция.
    :param data: исходный датафрейм.
    :param result_column: название признака, с которым работаем,
                          который получает изменения или заполнение.
    :param applying_column: название признака, по которому работает функция.
    :return: датафрейм с измененным признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    assert isinstance(result_column, str), 'Проблема с данными на входе, ' \
                                           'result_column должен быть str.'
    assert isinstance(applying_column, str), 'Проблема с данными на входе, ' \
                                             'applying_column должен быть str.'
    data.loc[data[result_column].isna(), result_column] = \
        data.loc[data[result_column].isna(), applying_column].apply(function)
    return data


def anomaly_engine_volume(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Замена выбросов в признаке Объём двигателя.
    :param data: исходный датафрейм.
    :return: датафрейм с измененным признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    if data.loc[data['Объём_двигателя_л'] > 8, 'Объём_двигателя_л'].any():
        data.loc[data['Объём_двигателя_л'] > 8, 'Объём_двигателя_л'] = np.NaN
    else:
        pass
    return data


def clean_climate(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция очистки признака Управление климатом от мусорных значений.
    :param data: исходный датафрейм.
    :return: очищенный датафрейм.
    '''
    mode_list = ['Климат-контроль многозонный', 'Кондиционер',
                 'Климат-контроль однозонный', 'Управление на руле',
                 'Атермальное остекление']
    data.loc[~data['Управление_климатом'].isin(mode_list),
             'Управление_климатом'] = 'Климат-контроль многозонный'
    return data


def column_mapper(data: pd.DataFrame, column_to_bin: str,
                  bins_dict: dict) -> pd.DataFrame:
    '''
    Замена значений в признаке по словарю.
    :param data: исходный датафрейм.
    :param column_to_bin: название признака для замены.
    :param bins_dict: словарь для замены значений.
    :return: датафрейм с бинаризованным признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    data[column_to_bin] = data[column_to_bin].map(bins_dict)
    return data


def data_drop_poor_categories(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Удаление объектов, имеющих всего 1 категорию в признаке Модификация.
    :param data: исходный датафрейм.
    :return: датафрейм с заполненным признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    poor_categories = data['Модификация'].value_counts()[
                                                data['Модификация'].value_counts() < 2].index
    if len(poor_categories) > 0:
        data = data.loc[~data['Модификация'].isin(poor_categories)]
    else:
        pass
    return data


def many_owners_gen(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Генерация признака Много владельцев.
    :param data: исходный датафрейм.
    :return: датафрейм с новым признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    data['Много_владельцев'] = np.where(data['Владельцев_по_ПТС'] >= 4, 1, 0)
    return data


def service_life_gen(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Генерация признака Срок эксплуатации.
    :param data: исходный датафрейм.
    :return: датафрейм с новым признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    curr_year = datetime.now().year
    data['Срок_эксплуатации'] = curr_year - data['Год_выпуска']
    return data


def deterioration_bins_gen(data: pd.DataFrame,
                           bins: list, labels: list) -> pd.DataFrame:
    '''
    Генерация признака степень износа через формирование бинов по признаку Пробег.
    :param data: исходный датафрейм.
    :param bins: список с числовыми границами бинов.
    :param labels: наименование бинов.
    :return: датафрейм с новым признаком.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    data['Степень_износа'] = pd.cut(
        data['Пробег_км'],
        bins=bins,
        labels=labels)
    return data


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Переименование названий признаков в датафрейме,
    чтобы убрать мусорные символы и пробелы.
    :param data: исходный датафрейм.
    :return: датафрейм с переименованными признаками.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    data = data.rename(columns=lambda x: re.sub('[^A-Za-zА-яа-я0-9_-ё]+', '_', x))
    data['Поколение'] = data['Наименование_поколения']
    return data


def column_drop(data: pd.DataFrame, columns_list: list) -> pd.DataFrame:
    '''
    Удаление колонок.
    :param data: исходный датафрейм.
    :param columns_list: список с именами прознаков.
    :return: датафрейм без некоторых признаков.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    assert isinstance(columns_list, list), 'Проблема с данными на входе, ' \
                                           'columns_list должен быть list.'
    data = data.drop(columns_list, axis=1)
    return data


def data_fillna_train_test(data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Заполнение пропусков в датасете, предварительно разделяя его на train и test.
    :param data: датасет.
    :return: train/test датасеты с заполненными данными.
    '''
    cat_columns = data.select_dtypes(include='object').columns
    num_columns = data.select_dtypes(exclude='object').columns
    strat_column = data[cat_columns].nunique().idxmin()
    df_train, df_test = split_dataset_holdout(dataset=data,
                                              stratify=data[strat_column],
                                              **kwargs)

    numeric_transformer = Pipeline(
        steps=[('imputer',
                IterativeImputer(estimator=RandomForestRegressor(
                    random_state=kwargs['random_state']),
                    max_iter=10,
                    random_state=kwargs['random_state']))])

    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer,
                       num_columns), ('cat', categorical_transformer,
                                      cat_columns)])

    df_train = pd.DataFrame(preprocessor.fit_transform(df_train),
                            columns=num_columns.tolist() + cat_columns.tolist())
    df_test = pd.DataFrame(preprocessor.transform(df_test),
                           columns=num_columns.tolist() + cat_columns.tolist())
    for col in data.columns:
        df_train[col] = df_train[col].astype(data[col].dtype)
        df_test[col] = df_test[col].astype(data[col].dtype)

    return df_train, df_test


def save_unique_train_data(data: pd.DataFrame, drop_columns: list,
                              target_column: str, unique_values_path: str) -> None:
    '''
    Функция для сохранения уникальных наименований признаков.
    :param data: исходный датафрейм.
    :param drop_columns: список имен признаков для удаления.
    :param target_column: имя целевой переменной.
    :param unique_values_path: путь для сохранения.
    :return: не возвращает ничего.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    assert isinstance(drop_columns, list), 'Проблема с данными на входе, ' \
                                           'drop_columns должен быть list.'
    assert isinstance(target_column, str), 'Проблема с данными на входе, ' \
                                           'target_column должен быть str.'
    assert isinstance(unique_values_path, str), 'Проблема с данными на входе, ' \
                                                'unique_values_path должен быть str.'
    unique_data = data.drop(columns=drop_columns + [target_column], axis=1, errors='ignore')
    unique_dict = {key: unique_data[key].unique().tolist() for key in unique_data.columns}
    unique_dict['Год_выпуска'].sort()
    with open(unique_values_path, 'w', encoding='utf-8') as file:
        json.dump(unique_dict, file, ensure_ascii=False)


def save_cars_data(data: pd.DataFrame, cars_dict_path: str) -> None:
    '''
    Функция сохранения словаря с рейтингами автомобилей по их модификациям.
    :param data: датафрейм.
    :param cars_dict_path: путь до файла для сохранения.
    :return: None.
    '''
    assert isinstance(data, pd.DataFrame), 'Проблема с данными на входе, ' \
                                           'data должен быть pd.DataFrame.'
    group_columns = [
        'Бренд_авто', 'Модель_авто', 'Модификация', 'Поколение'
    ]
    data[group_columns] = data[group_columns].astype('str')
    grouped_df = data.groupby(group_columns)['Рейтинг'].agg('mean').round(2).reset_index(name='Рейтинг')
    brand_dict = {
        key: grouped_df[grouped_df['Бренд_авто'] == key]
        ['Модель_авто'].unique().tolist()
        for key in grouped_df['Бренд_авто'].unique()
    }
    model_dict = {
        key: grouped_df[grouped_df['Модель_авто'] == key]
        ['Модификация'].unique().tolist()
        for key in grouped_df['Модель_авто'].unique()
    }
    gen_dict = {
        key: grouped_df[grouped_df['Модификация'] == key]
        ['Поколение'].unique().tolist()
        for key in grouped_df['Модификация'].unique().tolist()
    }
    rating_dict = {
        key: grouped_df[grouped_df['Поколение'] == key]
        ['Рейтинг'].values[0]
        for key in grouped_df['Поколение'].unique().tolist()
    }
    cars_dict = {
        'Бренды': brand_dict,
        'Модели': model_dict,
        'Поколения': gen_dict,
        'Рейтинги': rating_dict
    }
    data[group_columns] = data[group_columns].astype('category')
    with open(cars_dict_path, 'w', encoding='utf-8') as file:
        json.dump(cars_dict, file, ensure_ascii=False)



def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    '''
    Проверка на соответствие признаков в train и test данных и их упорядочивание.
    :param data: тестовый датафрейм.
    :param unique_values_path: путь до файла с уникальными значениями.
    :return: тестовый датафрейм, прошедший проверку и упорядоченный.
    '''
    with open(unique_values_path, 'r', encoding='utf-8') as json_file:
        unique_values = json.load(json_file)
    columns_seq = unique_values.keys()
    assert set(data.columns) == set(columns_seq), 'Проверьте признаки в ' \
                                                  'тренировочных и тестовых данных.'
    return data[columns_seq]


def preprocessing_pipeline(data: pd.DataFrame, flg_evaluate: bool = True,
                           save_csv=False, **kwargs) -> \
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    '''
    Конвеер для препроцессинга данных для трансформирования их в удобный рабочий вид.
    :param data: исходный датафрейм.
    :param flg_evaluate: флаг проверки соответствия признаков.
    :param kwargs: дополнительные параметры из конфигурации.
    :return: два трансформированных датафрейма train/test.
    '''
    if (data.isna().sum().sum() > 0) or ('Название авто' in data.columns):
        data = rename_columns(data=data)
        data = apply_function(function=modification_fill,
                              data=data,
                              result_column='Модификация',
                              applying_column='Название_авто')
        data = anomaly_engine_volume(data=data)
        data = clean_climate(data=data)
        data = column_drop(data=data, columns_list=kwargs['drop_columns'])
        data = data_drop_poor_categories(data=data)
        df_train, df_test = data_fillna_train_test(data=data, **kwargs)
        for dataframe in (df_train, df_test):
            dataframe = column_mapper(data=dataframe,
                                      column_to_bin='Обмен',
                                      bins_dict=kwargs['map_change_columns']['Обмен'])
            dataframe = many_owners_gen(data=dataframe)
            dataframe = service_life_gen(data=dataframe)
            dataframe = deterioration_bins_gen(data=dataframe,
                                               bins=kwargs['map_bins']['Пробег_км']['bins'],
                                               labels=kwargs['map_bins']['Пробег_км']['labels'])
            dataframe = transform_types(data=dataframe,
                                        types_dict=kwargs['change_type_columns'],
                                        err='raise',
                                        object_to_category=True)
        if save_csv:
            df_train.to_csv(kwargs['train_path_proc'], index=False)
            df_test.to_csv(kwargs['test_path_proc'], index=False)
    else:
        data = transform_types(data=data,
                               types_dict=kwargs['change_type_columns'],
                               err='raise',
                               object_to_category=True)
    if flg_evaluate:
        check_columns_evaluate(data=data, unique_values_path=kwargs['unique_values_path'])
        return data
    else:
        save_unique_train_data(data=df_train,
                               drop_columns=kwargs['drop_columns'],
                               target_column=kwargs['target_column'],
                               unique_values_path=kwargs['unique_values_path'])
        save_cars_data(data=data, cars_dict_path=kwargs['cars_dict_path'])
        return df_train, df_test
