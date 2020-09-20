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


def load_big_data():

    df = pd.read_csv('stacked_total.csv', delimiter=';', encoding='utf-8')

    df_by_years = []

    for i in range(1, 5):

        df_by_years.append(df.loc[df['year'] == i])


    # training data
    training_data = []
    training_data_values = []
    test_data = []
    test_data_values = []
    training_inputs = []
    test_inputs = []

    for df in df_by_years:

        test = df.sample(frac=0.25)

        training = df.merge(test, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']

        new_indexes = [column for column in
                       training.columns.difference(['_merge'], sort=False)]

        training = training[new_indexes]

        training_data.append(training)
        test_data.append(test)



    for i in range(len(training_data)):

        df_year = df_by_years[i]
        training = training_data[i]
        test = test_data[i]

        weeks = None

        if i == 3:
            weeks = 29

        else:
            weeks = 38

        check = df_year.loc[df_year['week'] == weeks][['first_name','total_points','second_name']]

        answers1 = training.merge(check, on=('second_name', 'first_name'), how='inner')
        answers2 = test.merge(check, on=('second_name', 'first_name'), how='inner')
        training = answers1
        test = answers2
        new_indexes = [column for column in answers1.columns.difference(['first_name', 'second_name', 'year',
                                                                                 'Unnamed: 0', '_merge',
                                                                                 'total_points_y']
                                                                                , sort=False)]
        training_data_values.append(np.resize(answers1['total_points_y'].to_numpy(),(answers1['total_points_y'].to_numpy().size,1)))
        test_data_values.append(np.resize(answers2['total_points_y'].to_numpy(), (answers2['total_points_y'].to_numpy().size,1)))

        test_inputs.append(test[new_indexes].to_numpy())
        training_inputs.append(training[new_indexes].to_numpy())


    training_inputs = np.concatenate(training_inputs,axis=0)
    training_data_values = np.concatenate(training_data_values,axis=0)
    test_inputs = np.concatenate(test_inputs,axis=0)
    test_data_values = np.concatenate(test_data_values, axis=0)



    return  training_inputs.astype(dtype=float), training_data_values.astype(dtype=float), test_inputs, test_data_values
















































