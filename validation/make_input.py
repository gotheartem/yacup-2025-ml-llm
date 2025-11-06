import json
import pandas as pd

test = pd.read_csv('validation/test.csv')

with open('validation/input.json', 'w') as file:
    json.dump(test['query'].tolist(), file)