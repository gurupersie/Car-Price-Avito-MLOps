'''
Программа обучения модели на backend, отображение метрик и графиков обучения на экране.
Версия: 0.1.
'''

import os
import json
import joblib
import requests
import streamlit as st
from millify import millify
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    '''
    Обучение модели с выводом результатов.
    :param config: конфигурационный файл.
    :param endpoint: endpoint.
    '''
    assert isinstance(config, dict), 'Проверьте config, должен быть dict.'
    assert isinstance(endpoint, object), 'Проверьте endpoint, должен быть object.'
    if os.path.exists(config['train']['metrics_path']):
        with open(config['train']['metrics_path'], 'r', encoding='utf-8') as json_file:
            old_metrics = json.load(json_file)
    else:
        old_metrics = {'MAE': 0,
                       'R2_Adjusted': 0,
                       'WAPE': 0,
                       'RMSE': 0,
                       'MAE diff, %': 0}

    with st.spinner('Модель подбирает параметры...'):
        output = requests.post(endpoint, timeout=8000)
    st.success('Успешно!')

    new_metrics = output.json()['metrics']

    mae, r2_adjusted, wape, rmse, mae_diff = st.columns(5)

    mae.metric('MAE',
               millify(new_metrics['MAE']),
               f'{new_metrics["MAE"] - old_metrics["MAE"]:.2f}')
    r2_adjusted.metric('R2_Adjusted',
                       round(new_metrics['R2_Adjusted'], 2),
                       f'{new_metrics["R2_Adjusted"] - old_metrics["R2_Adjusted"]:.2f}')
    wape.metric('WAPE',
                millify(new_metrics['WAPE']),
                f'{new_metrics["WAPE"] - old_metrics["WAPE"]:.2f}')
    rmse.metric('RMSE',
                millify(new_metrics['RMSE']),
                f'{new_metrics["RMSE"] - old_metrics["RMSE"]:.2f}')
    mae_diff.metric('MAE test/train',
                    millify(new_metrics['MAE diff, %']),
                    f'{new_metrics["MAE diff, %"] - old_metrics["MAE diff, %"]:.2f}')

    study = joblib.load(os.path.join(config['train']['study_path']))
    fig_importances = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_importances, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
