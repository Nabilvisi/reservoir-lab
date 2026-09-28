"""Emit only release files for authenticated connector publishing; exclude local state."""
from pathlib import Path
import base64
import json

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / name for name in [
    '.gitignore', '.gitattributes', '.streamlit/config.toml', 'app.py', 'launch.ps1', 'LICENSE',
    'METHODS.md', 'README.md', 'requirements.txt', 'requirements-lock.txt', 'VALIDATION.md',
    'data/provenance.json', 'data/spe10_subset.npz', 'data/training.csv',
    'data/training_provenance.json', 'data/validation.json', 'data/raw/odbl-10.txt',
    'data/raw/dbcl-10.txt', 'data/raw/README.md', 'data/raw/SPE10_MODEL2.DATA',
    'artifacts/reservoir-preview.png']]
for folder in ['reservoir', 'tests', 'scripts']:
    paths.extend(sorted((ROOT / folder).glob('*.py')))
files = []
for path in paths:
    binary = path.suffix in {'.png', '.npz'}
    files.append({'path': path.relative_to(ROOT).as_posix(),
                  'encoding': 'base64' if binary else 'utf-8',
                  'content': base64.b64encode(path.read_bytes()).decode() if binary else path.read_bytes().decode('utf-8')})
print(json.dumps(files))
