"""Run the backend's /api/preprocess endpoint using Flask test client.
This runs inside the same Python environment without network calls and prints the JSON response.
"""
import os, sys
# Ensure backend root is on sys.path so we can import app when running from scripts/
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app

with app.test_client() as client:
    resp = client.post('/api/preprocess')
    print('Status code:', resp.status_code)
    try:
        print('JSON:', resp.get_json())
    except Exception:
        print('Response data:', resp.data.decode())
