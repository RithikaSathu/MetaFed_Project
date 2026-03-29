import requests, os
fpath = os.path.join(os.path.dirname(__file__), '..', 'assets', 'sample_image.png')
with open(fpath,'rb') as f:
    files = {'file': ('sample_image.png', f, 'image/png')}
    data = {'algorithm':'metafed_hom'}
    r = requests.post('http://127.0.0.1:5000/api/image-upload', files=files, data=data, timeout=30)
    print('Status', r.status_code)
    print(r.json())

with open(fpath,'rb') as f:
    files = {'file': ('sample_image.png', f, 'image/png')}
    data = {'algorithm':'metafed_het'}
    r2 = requests.post('http://127.0.0.1:5000/api/image-upload', files=files, data=data, timeout=30)
    print('Status', r2.status_code)
    print(r2.json())
