"""
Программа: Отрисовка графиков.
Версия: 0.1.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def plot_bars(df: pd.DataFrame,
              target: str,
              feature: str,
              ax: plt.Axes = None) -> plt.Figure:
    '''
    Функция отрисовки графика barplot с долями и подписями в процентах.
    :param df: pd.DataFrame - исходный датасет с данными.
    :param target: str - наименовение целевой переменной, по которой группируем.
    :param feature: str - наименование колонки-признака, которую рассматривае.
    :param ax: np.array - подграфик фигуры.
    :return: график barplot.
    '''
    assert isinstance(df, pd.DataFrame), "Неверный тип данных на входе, " \
                                         "df должен быть pd.DataFrame."
    assert isinstance(target, str), "Неверный тип данных на входе, " \
                                    "target должен быть str."
    assert isinstance(feature, str), "Неверный тип данных на входе, " \
                                     "feature должен быть str."

    normed_groups = df.groupby(target)[feature].value_counts(
        normalize=True).mul(100).rename('percent').reset_index()

    if ax is None:
        figure, ax = plt.subplots(figsize=(15, 7))
    else:
        figure = ax.get_figure()

    sns.barplot(normed_groups,
                x=target,
                y='percent',
                palette='bright',
                hue=feature,
                ax=ax,
                legend='brief')

    for p in ax.patches[:-df[feature].nunique()]:
        percentage = f'{p.get_height():.1f} %'
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center',
                    va='center',
                    textcoords='offset points',
                    xytext=(0, 20),
                    rotation=90,
                    fontsize=6)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title(feature, fontsize=18)
    ax.set_xlabel('Классы', fontsize=14)
    ax.set_ylabel('Проценты', fontsize=14)
    ax.set_ylim(0, 115)
    ax.set_xticklabels(df[target].unique(), rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    return figure


def plot_countplot(df: pd.DataFrame, feature_name: str) -> matplotlib.figure.Figure:
    '''
    Функция отрисовки графика countplot с процентными долями.
    :param df: pd.DataFrame - датафрейм с данными.
    :param column_name: str - Название признака.
    :return: график countplot.
    '''

    assert isinstance(df, pd.DataFrame), "Неверный тип данных на входе, " \
                                         "df должен быть pd.DataFrame."
    assert isinstance(feature_name, str), "Неверный тип данных на входе, " \
                                          "feature_name должен быть str."

    figure = plt.figure(figsize=(15, 8))

    ax = sns.countplot(data=df[feature_name],
                       palette='Set1',
                       order=df[feature_name].value_counts().index)
    total = df[feature_name].count()

    for container in ax.containers:
        ax.bar_label(container, fmt=lambda x: f'{(x/total)*100:0.1f}%')

    plt.title(f'График количества элементов в признаке {feature_name}', fontsize=20)
    plt.xlabel('Количество', fontsize=14)
    plt.ylabel(f'{feature_name}', fontsize=14)
    return figure


def plot_median_barplot(df: pd.DataFrame, group_feature: str, target_feature: str
                        ) -> matplotlib.figure.Figure:
    '''
    Функция отрисовки графика barplot.
    :param group_feature: str - Название признака, по которому группируем.
    :param target_feature: str - Название признака целевого.
    :return: график barplot.
    '''

    assert isinstance(df, pd.DataFrame), "Неверный тип данных на входе, " \
                                         "df должен быть pd.DataFrame."
    assert isinstance(group_feature, str), "Неверный тип данных на входе, " \
                                           "group_feature должен быть str."
    assert isinstance(target_feature, str), "Неверный тип данных на входе, " \
                                            "target_feature должен быть str."

    figure, ax = plt.subplots(figsize=(8, 10))

    ax = sns.barplot(x=df.groupby(group_feature)[target_feature].median().values,
                y=df.groupby(group_feature)[target_feature].median().index,
                order=df.groupby(group_feature)[target_feature].median().sort_values(
                    ascending=False).index,
                palette='bright',
                ax=ax)
    for cont in ax.containers:
        ax.bar_label(cont, fmt=lambda x: f'{x:.0f}')
    ax.margins(x=0.15)
    plt.title(f'Медианное значение признака {target_feature} '
              f'в разрезе {group_feature}', fontsize=20)
    return figure


def corr_matrix(data: pd.DataFrame) -> None:
    '''
    Функция отрисовки матрицы корреляций.
    :param data: исходный датафрейм
    '''
    assert isinstance(data, pd.DataFrame), "Неверный тип данных на входе, " \
                                           "data должен быть pd.DataFrame."
    figure = plt.figure(figsize=(15, 11))
    sns.heatmap(data.select_dtypes(exclude='object').corr(), cmap='coolwarm', annot=True)
    plt.title('Корреляция признаков.', fontsize=20)
    return figure
