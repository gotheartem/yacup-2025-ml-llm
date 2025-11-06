import json
import pandas as pd

test = pd.read_csv('validation/test.csv').sample(n=32, random_state=42)

with open('validation/output.json', 'r') as file:
    test['answer'] = json.load(file)

print("SUCCESSFUL RUN")
print(test)