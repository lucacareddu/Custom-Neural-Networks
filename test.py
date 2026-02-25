import requests
import json

data = json.load(open('stream_test'))

pred = requests.post("http://localhost:8002/", json=data)

print(pred.json())