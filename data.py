import requests
import pandas as pd
import numpy as np

url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
json = r.json()
elements_types_df = pd.DataFrame(json['element_types'])


raw_data_frames = []
clean_data_frames = []

for i in range(1, 5):
    raw_data_frame = pd.read_csv('./data/' + str(i) +'_raw.csv', encoding='latin-1')
    clean_data_frame = pd.read_csv('./data/' + str(i) +'.csv', encoding='latin-1')
    raw_data_frames.append(raw_data_frame)
    clean_data_frames.append(clean_data_frame)


for i in range(len(raw_data_frames)):
    clean_data_frames[i]['position'] = raw_data_frames[i].element_type.map(elements_types_df.set_index('id').singular_name)

