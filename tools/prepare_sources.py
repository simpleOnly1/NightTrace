"""Populate NightTrace's own source tree once, without git or shadow copies.

This program imports only upstream DAV2 Python source and the official STEPS
split_files. STEPS data/evaluation algorithms are already adapted as native
NightTrace modules; this command never creates *original, external, or sibling
repositories.

Usage:
  python tools/prepare_sources.py
  python tools/prepare_sources.py --steps-zip /tmp/STEPS-master.zip \
     --dav2-zip /tmp/Depth-Anything-V2-main.zip
"""
import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'steps': 'https://codeload.github.com/ucaszyp/STEPS/zip/refs/heads/master',
    'dav2': 'https://codeload.github.com/DepthAnything/Depth-Anything-V2/zip/refs/heads/main',
}


def load_archive(name, zip_path=None, source_dir=None):
    if source_dir:
        directory = Path(source_dir)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        return {p.relative_to(directory).as_posix(): p.read_bytes()
                for p in directory.rglob('*') if p.is_file() and '.git' not in p.parts}
    if zip_path:
        blob = Path(zip_path).read_bytes()
    else:
        print('Downloading official %s source archive (no git)...' % name, flush=True)
        with urllib.request.urlopen(URLS[name], timeout=120) as fp:
            blob = fp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        result = {}
        for item in zf.infolist():
            if item.is_dir():
                continue
            chunks = Path(item.filename).parts
            if len(chunks) < 2 or '..' in chunks or item.file_size > 10_000_000:
                continue
            result[Path(*chunks[1:]).as_posix()] = zf.read(item)
        return result


def write(dest, contents, overwrite):
    dest = ROOT / dest
    if dest.exists() and dest.read_bytes() != contents and not overwrite:
        raise FileExistsError('%s differs; pass --force to replace' % dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps-zip')
    parser.add_argument('--dav2-zip')
    parser.add_argument('--steps-dir', help='Existing extracted STEPS source directory')
    parser.add_argument('--dav2-dir', help='Existing extracted DAV2 source directory')
    parser.add_argument('--dav2-only', action='store_true')
    parser.add_argument('--splits-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.dav2_only and args.splits_only:
        parser.error('--dav2-only and --splits-only are mutually exclusive')
    if (args.steps_zip and args.steps_dir) or (args.dav2_zip and args.dav2_dir):
        parser.error('Choose either a ZIP or a directory for each source')
    receipt = {}
    if not args.dav2_only:
        entries = load_archive('steps', args.steps_zip, args.steps_dir)
        splits = [k for k in entries if k.startswith('split_files/') and not k.endswith('/')]
        if not splits:
            raise RuntimeError('Official STEPS archive contains no split_files/')
        for rel in splits:
            if Path(rel).suffix.lower() not in ('.txt', '.json', '.csv', '.md', '.yaml'):
                continue
            receipt[rel] = write(rel, entries[rel], args.force)
        # Store upstream source digest in the manifest without generating a shadow repository.
        for rel in ('test_robotcar_disp.py', 'test_nuscenes_disp.py',
                    'evaluation/eval_nuscenes.py', 'evaluation/eval_robotcar.py',
                    'show_eval_result.py'):
            if rel in entries:
                receipt['upstream_source_sha256:'+rel] = hashlib.sha256(entries[rel]).hexdigest()
        print('Imported %d STEPS split files' % len(splits))
    if not args.splits_only:
        entries = load_archive('dav2', args.dav2_zip, args.dav2_dir)
        source = [k for k in entries if k.startswith('depth_anything_v2/') and k.endswith('.py')]
        if not any(k.endswith('/dpt.py') for k in source):
            raise RuntimeError('Official DAV2 dpt.py missing')
        for rel in source:
            target = 'models/' + rel
            receipt[target] = write(target, entries[rel], args.force)
        if 'LICENSE' in entries:
            receipt['licenses/DAV2_LICENSE'] = write('licenses/DAV2_LICENSE', entries['LICENSE'], args.force)
        print('Imported %d DAV2 source files into models/depth_anything_v2/' % len(source))
    path = ROOT / 'SOURCE_MANIFEST.json'
    past = json.loads(path.read_text()) if path.exists() else {}
    past.update(receipt)
    path.write_text(json.dumps(past, indent=2, sort_keys=True))
    print('Done. All runtime code stays inside', ROOT)


if __name__ == '__main__':
    main()
