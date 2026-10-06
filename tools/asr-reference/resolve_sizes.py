import concurrent.futures, json, sys, urllib.parse, urllib.request
from pathlib import Path
source, target = map(Path, sys.argv[1:3])
report = json.loads(source.read_text(encoding='utf-8-sig'))
def size(item):
    url = urllib.parse.urldefrag(item['download_info']['url'])[0]
    name, version = item['metadata']['name'], item['metadata']['version']
    if name.lower() == 'torch' and '+cpu' in version:
        import hashlib
        expected = item['download_info']['archive_info']['hashes']['sha256']
        candidates = Path('tools/asr-reference/pip-cache').rglob('*.body')
        length = None
        for candidate in candidates:
            if candidate.stat().st_size < 100_000_000:
                continue
            with candidate.open('rb') as file:
                digest = hashlib.file_digest(file, 'sha256').hexdigest()
            if digest == expected:
                length = candidate.stat().st_size
                break
        if length is None:
            raise RuntimeError('Official CPU torch wheel hash not found in workspace cache')
    else:
        request = f'https://pypi.org/pypi/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/json'
        with urllib.request.urlopen(request, timeout=40) as response:
            metadata = json.load(response)
        filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit('/', 1)[-1])
        length = next(x['size'] for x in metadata['urls'] if x['filename'] == filename)
    return {'name': item['metadata']['name'], 'version': item['metadata']['version'], 'bytes': length, 'url': url}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    files = list(executor.map(size, report['install']))
result = {'packages': files, 'totalWheelBytes': sum(x['bytes'] for x in files), 'packageCount': len(files)}
target.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'totalWheelBytes': result['totalWheelBytes'], 'packageCount': result['packageCount']}))
