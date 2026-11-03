"""Compatibility contracts; the pure source tests also run on Python 3.8."""
import ast
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.unidepth_compat import backport_source, PATCH_VERSION
from tools.prepare_unidepth import import_archive


class StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == 'typing' and all(a.asname and a.asname.startswith('_NT_') for a in node.names):
            return None
        return node


class CompatibilityTests(unittest.TestCase):
    def test_backport_evaluates_annotations_and_preserves_computation(self):
        source = '''"""模块文档"""
from __future__ import division
def f(x: list[tuple[int, str | None]], y: dict[str, list[int]] = None) -> int | None:
    text = "中文"; values: list[int] = [a for a, b in x if b is not None]
    return sum(values) if values else None
'''
        result, changes = backport_source(source, 'example.py')
        scope = {}
        exec(result, scope)
        self.assertEqual(scope['f']([(1, 'ok'), (4, None), (2, 'ok')]), 3)
        self.assertEqual(changes['annotations'], 4)
        self.assertEqual(scope['__doc__'], '模块文档')
        self.assertIn('_NT_List[_NT_Tuple[int, _NT_Union[str, None]]]', result)
        self.assertEqual(backport_source(result, 'example.py')[0], result)
        # Remove annotations to check executable statements, including operators.
        class StripAll(StripAnnotations):
            def visit_AnnAssign(self, node):
                return ast.Assign(targets=[node.target], value=node.value)
        self.assertEqual(ast.dump(StripAll().visit(ast.parse(source))),
                         ast.dump(StripAll().visit(ast.parse(result))))

    def test_only_optional_visualization_reexport_is_removed(self):
        source = 'from .camera import Camera\nfrom .visualization import colorize\nx = 2 * 3\n'
        result, changes = backport_source(source, 'unidepth/utils/__init__.py')
        self.assertTrue(changes['optional_visualization'])
        self.assertIn('from .camera import Camera', result)
        self.assertIn('x = 2 * 3', result)
        self.assertNotIn('import colorize', result)
        self.assertEqual(backport_source(source, 'other.py')[0], source)

    def test_existing_exact_upstream_source_can_be_patched_without_force(self):
        source = 'def f(x: list[int]) -> int | None:\n    return sum(x)\n'
        rel = 'models/unidepth/models/unidepthv2/unidepthv2.py'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('upstream/' + rel[len('models/'):], source)
            # The official ZIP includes a 17 MB documentation GIF, not source.
            ignored = zipfile.ZipInfo('upstream/assets/demo.gif')
            ignored.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(ignored, b'0' * 10000001)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/rel
            path.parent.mkdir(parents=True)
            path.write_bytes(source.encode('utf-8'))
            receipt = import_archive(buffer.getvalue(), root)
            self.assertEqual(receipt['compatibility']['version'], PATCH_VERSION)
            self.assertIn(rel, receipt['compatibility']['original_sha256'])
            self.assertIn('_NT_List[int]', path.read_text(encoding='utf-8'))
            self.assertEqual(import_archive(buffer.getvalue(), root), receipt)
            path.write_text('# user edits', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                import_archive(buffer.getvalue(), root)

    def test_requirements_exclude_existing_numerical_stack(self):
        requirements = (Path(__file__).resolve().parents[1]/'requirements-unidepth.txt').read_text()
        names = {line.strip().split('==')[0].lower() for line in requirements.splitlines()
                 if line.strip() and not line.lstrip().startswith('#')}
        self.assertFalse(names & {'torch', 'torchvision', 'numpy', 'scipy', 'opencv-python', 'pillow'})

    def test_backported_annotations_work_with_torchscript(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is not installed in this source-only Python runtime')
        source = '''import torch
@torch.jit.script
def f(xs: list[torch.Tensor]) -> torch.Tensor:
    result = torch.zeros_like(xs[0])
    for x in xs:
        result = result + x
    return result
'''
        result, _ = backport_source(source, 'jit_fixture.py')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'jit_fixture.py'; path.write_text(result)
            spec = importlib.util.spec_from_file_location('nt_jit_fixture', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                self.assertTrue(torch.equal(module.f([torch.ones(2), torch.ones(2)*2]), torch.ones(2)*3))
            finally:
                sys.modules.pop(spec.name, None)


if __name__ == '__main__':
    unittest.main()
