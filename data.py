import requests
import pandas as pd
import numpy as np


def preprocces_data():

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()
    elements_types_df = pd.DataFrame(json['element_types'])


    raw_data_frames = []
    clean_data_frames = []
    inputs = []
    outputs = []
    for i in range(1, 6):
        raw_data_frame = pd.read_csv('./data/' + str(i) +'_raw.csv', encoding='latin-1')
        clean_data_frame = pd.read_csv('./data/' + str(i) +'.csv', encoding='latin-1')
        raw_data_frames.append(raw_data_frame)
        clean_data_frames.append(clean_data_frame)

    for i in range(len(raw_data_frames)):
        df = clean_data_frames[i]
        df['position'] = raw_data_frames[i].element_type.map(elements_types_df.set_index('id').singular_name)
        if i > 0:
            df.drop(columns=['now_cost'], inplace=True)

        if i == 4:
            df.drop(columns=['element_type'], inplace=True)

        df['goalie'] = 0
        df['def'] = 0
        df['mid'] = 0
        df['att'] = 0

        df.loc[(df['position'] == 'Goalkeeper'), 'goalie'] = 1
        df.loc[(df['position'] == 'Defender'), 'def'] = 1
        df.loc[(df['position'] == 'Midfielder'), 'mid'] = 1
        df.loc[(df['position'] == 'Forward'), 'att'] = 1

    new_indexes = [column for column in clean_data_frames[0].columns.difference(['first_name', 'second_name','position'],sort=False)]

    # keep only the players that are present 4 years
    merged = None
    for i in range(3):

        if merged is None:
            df_inputs = clean_data_frames[0]
            df_outputs = clean_data_frames[1]

            merged = pd.merge(df_inputs, df_outputs[['first_name', 'second_name']])
            continue

        else:

            df_inputs = clean_data_frames[i+1]

            merged = pd.merge(merged, df_inputs[['first_name', 'second_name']])


    for i in range(3):

        df = clean_data_frames[i]

        eliminated = pd.merge(df, merged[['first_name', 'second_name']])
        inputs.append(eliminated[new_indexes].to_numpy())


    eliminated_outputs = pd.merge(clean_data_frames[3], merged[['first_name', 'second_name']])


    features = np.concatenate(tuple(inputs), axis=1)
    y_actual = eliminated_outputs['total_points'].to_numpy()

    train_indexes = np.arange(160)
    test_indexes = np.arange(160, 190)
    val_indexes = np.arange(190,211)

    return features, np.resize(y_actual, (y_actual.size, 1)), train_indexes, test_indexes, val_indexes


















