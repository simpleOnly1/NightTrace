"""Import pinned official UniDepth source into models/; never download weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
if __package__:
    from .unidepth_compat import backport_source, PATCH_VERSION
else:
    from unidepth_compat import backport_source, PATCH_VERSION


ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8d8cfe4c7ee15297099983607febf0d4f32eb3d6'
URL = 'https://codeload.github.com/lpiccinelli-eth/UniDepth/zip/' + COMMIT


def import_archive(blob, root=ROOT, force=False):
    root = Path(root).resolve()
    files = {}
    originals = {}
    patches = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for item in archive.infolist():
            parts = item.filename.split('/')
            if item.is_dir() or len(parts) < 2:
                continue
            rel = '/'.join(parts[1:])
            if any(p in ('..', '') for p in parts):
                raise ValueError('Unsafe source entry: ' + item.filename)
            if rel.startswith('unidepth/') and (rel.endswith('.py') or rel.endswith('.so')):
                if not rel.endswith('.py'):
                    continue  # no platform-specific binary extensions
                target = root / 'models' / rel
            elif rel == 'requirements.txt':
                target = root / 'models/unidepth_upstream_requirements.txt'
            elif rel == 'LICENSE':
                target = root / 'licenses/UNIDEPTH_LICENSE'
            else:
                continue
            if item.file_size > 10000000:
                raise ValueError('Oversized source entry: ' + item.filename)
            try:
                target.resolve().relative_to(root)
            except ValueError:
                raise ValueError('Source escapes project')
            original = archive.read(item)
            originals[target] = original
            content = original
            if rel.endswith('.py'):
                patched, changes = backport_source(original.decode('utf-8'), rel)
                content = patched.encode('utf-8')
                if any(changes.values()):
                    patches[rel] = changes
            files[target] = content
    required = root / 'models/unidepth/models/unidepthv2/unidepthv2.py'
    if required not in files:
        raise ValueError('Archive does not contain official UniDepth V2 inference source')
    # Preflight all collisions before writing any source.
    for path, content in files.items():
        if path.exists() and path.read_bytes() not in (content, originals[path]) and not force:
            raise FileExistsError('Source differs: ' + str(path) + '; use a clean destination or --force')
    hashes = {}
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    receipt = dict(repository='lpiccinelli-eth/UniDepth', expected_commit=COMMIT,
                   archive_sha256=hashlib.sha256(blob).hexdigest(), files=hashes,
                   compatibility=dict(version=PATCH_VERSION, changed_files=patches,
                                      original_sha256={
                                          p.relative_to(root).as_posix(): hashlib.sha256(b).hexdigest()
                                          for p,b in originals.items()}),
                   note='Local ZIP content is hashed; filename alone is not commit attestation')
    (root / 'UNIDEPTH_SOURCE_MANIFEST.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', help='Offline ZIP of the pinned official commit')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.zip:
        blob = Path(args.zip).read_bytes()
    else:
        print('Downloading source only:', URL, flush=True)
        with urllib.request.urlopen(URL, timeout=120) as stream:
            blob = stream.read()
    receipt = import_archive(blob, force=args.force)
    print('Imported', len(receipt['files']), 'files. Weights remain a separate download.')


if __name__ == '__main__':
    main()
