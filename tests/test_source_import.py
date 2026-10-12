"""One-time source-import contract. No network; no source shadow dirs."""
from pathlib import Path
import importlib.util
import tempfile

source = Path(__file__).resolve().parents[1] / 'tools' / 'prepare_sources.py'
spec = importlib.util.spec_from_file_location('nighttrace_sources', str(source))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as temp:
    module.ROOT = Path(temp)
    digest = module.write('split_files/demo.txt', b'one\ntwo\n', False)
    assert (Path(temp)/'split_files/demo.txt').read_bytes() == b'one\ntwo\n'
    assert len(digest) == 64
    try:
        module.write('split_files/demo.txt', b'different', False)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Overwrites should fail')
    module.write('models/depth_anything_v2/dpt.py', b'class Example: pass\n', False)
    assert (Path(temp)/'models/depth_anything_v2/dpt.py').exists()
    assert not any('original' in p.name.lower() for p in Path(temp).rglob('*'))
print('PASS: native-only one-time source import and overwrite protection')
