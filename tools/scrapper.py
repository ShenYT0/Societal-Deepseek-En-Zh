import json
with open('../data/DeepSeek-top-100-results.json', 'r') as f:
    data = json.load(f)
    print(data)