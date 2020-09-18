import requests
import pandas as pd
import numpy as np


# API URL
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

# Use the requests package to make a GET request from the API endpoint
r = requests.get(url)
import collections
# 
json = r.json()
