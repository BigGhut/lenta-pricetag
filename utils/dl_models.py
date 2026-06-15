import os
import urllib.request
import tarfile

models = [
    ('det', 'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'),
    ('rec', 'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ru_PP-OCRv3_rec_infer.tar'),
]
home = os.path.expanduser('~')
base = os.path.join(home, '.paddleocr', 'whl')

for mtype, url in models:
    tar_name = url.split('/')[-1].replace('.tar', '')
    if mtype == 'det':
        target = os.path.join(base, 'det', 'ml', tar_name)
    else:
        target = os.path.join(base, 'rec', 'ru', tar_name)

    model_file = os.path.join(target, 'model.pdmodel')
    if os.path.exists(model_file):
        print(f'{mtype}: already exists at {target}')
        continue

    host = url.split('/')[2]
    print(f'{mtype}: downloading from {host} ...')
    tar_path = os.path.join(os.environ.get('TEMP', '/tmp'), tar_name + '.tar')
    try:
        # Контекст SSL применяется только к этому запросу, а не ко всему процессу.
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlretrieve(url, tar_path, context=ctx)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with tarfile.open(tar_path) as t:
            t.extractall(path=os.path.dirname(target))
        os.remove(tar_path)
        total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(target) for f in fn)
        print(f'  OK: {total/1024:.0f} KB')
    except Exception as e:
        print(f'  FAILED: {e}')
