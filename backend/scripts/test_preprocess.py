import requests

try:
    r = requests.post('http://127.0.0.1:5000/api/preprocess', timeout=60)
    print('Status:', r.status_code)
    print('Body:', r.text)
except Exception as e:
    print('Error calling preprocess:', e)
