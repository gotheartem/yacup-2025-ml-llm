import json
import pandas as pd

test = pd.read_csv('validation/test.csv').sample(n=32, random_state=42)

with open('validation/input.json', 'w') as file:
    json.dump(test['query'].tolist(), file)