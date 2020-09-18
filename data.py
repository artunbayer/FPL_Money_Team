import requests
import pandas as pd
import numpy as np


# API URL
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

# Use the requests package to make a GET request from the API endpoint
r = requests.get(url)

# transform that request into a json object:
json = r.json()
# pushs
events_df = pd.DataFrame(json['elements'])
print(events_df)

