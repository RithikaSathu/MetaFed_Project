"""Emulate frontend call to /api/preprocess (sets Origin header) and then GET /api/health to show updated status."""
import requests

BASE = "http://127.0.0.1:5000"
HEADERS = {"Origin": "http://localhost:8080"}

print('Calling POST /api/preprocess as if from frontend (Origin http://localhost:8080)')
r = requests.post(f"{BASE}/api/preprocess", headers=HEADERS, timeout=120)
print('Preprocess response:', r.status_code, r.text)

print('\nQuerying /api/health to observe preprocessing status')
r2 = requests.get(f"{BASE}/api/health", headers=HEADERS, timeout=10)
print('Health response:', r2.status_code, r2.json())
