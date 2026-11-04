"""Python 3.8 annotation backport for a pinned UniDepth source snapshot.

Only annotation spans and an optional visualization re-export are changed.
Numerical operators, inference preprocessing and state-dict names are untouched.
"""
import ast

PATCH_VERSION = 'py38-annotations-v1'
ALIASES = {name: '_NT_' + name.title() for name in
           ('list', 'dict', 'tuple', 'set', 'frozenset', 'type')}
IMPORTS = ('\nfrom typing import (List as _NT_List, Dict as _NT_Dict, '
           'Tuple as _NT_Tuple, Set as _NT_Set, FrozenSet as _NT_Frozenset, '
           'Type as _NT_Type, Union as _NT_Union)\n')
# str.title() would produce Frozenset rather than typing.FrozenSet; the alias
# itself may have any name, but keep the spelling shared with IMPORTS.


def backport_source(source, relative_path):
    tree = ast.parse(source)
    lines = source.encode('utf-8').splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    raw = source.encode('utf-8')
    edits = []

    def bounds(node):
        return (offsets[node.lineno-1] + node.col_offset,
                offsets[node.end_lineno-1] + node.end_col_offset)

    def segment(node):
        start, end = bounds(node)
        return raw[start:end].decode('utf-8')

    def render(node):
        if isinstance(node, ast.Index):  # Python 3.8 AST
            return render(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return '_NT_Union[' + render(node.left) + ', ' + render(node.right) + ']'
        if isinstance(node, ast.Subscript):
            name = ALIASES.get(node.value.id) if isinstance(node.value, ast.Name) else None
            return (name or render(node.value)) + '[' + render(node.slice) + ']'
        if isinstance(node, ast.Tuple):
            return ', '.join(render(value) for value in node.elts)
        if isinstance(node, ast.List):
            return '[' + ', '.join(render(value) for value in node.elts) + ']'
        if isinstance(node, ast.Attribute):
            return render(node.value) + '.' + node.attr
        return segment(node)

    for node in ast.walk(tree):
        annotation = (node.annotation if isinstance(node, (ast.arg, ast.AnnAssign))
                      else node.returns if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                      else None)
        if annotation is None:
            continue
        modern = any(
            (isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr)) or
            (isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name)
             and part.value.id in ALIASES)
            for part in ast.walk(annotation))
        if modern:
            start, end = bounds(annotation)
            edits.append((start, end, render(annotation).encode('utf-8')))
    annotation_count = len(edits)
    if annotation_count:
        # Keep module docstrings and all existing future imports in legal order.
        insert_line = 0
        for i, node in enumerate(tree.body):
            if ((i == 0 and isinstance(node, ast.Expr) and
                 isinstance(node.value, (ast.Str, ast.Constant)) and
                 isinstance(getattr(node.value, 'value', getattr(node.value, 's', None)), str))
                    or (isinstance(node, ast.ImportFrom) and node.module == '__future__')):
                insert_line = node.end_lineno
            else:
                break
        edits.append((offsets[insert_line], offsets[insert_line], IMPORTS.encode('utf-8')))
    optional_visualization = False
    if relative_path == 'unidepth/utils/__init__.py':
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'visualization':
                start, end = bounds(node)
                edits.append((start, end, b'# Training/plotting re-exports omitted for frozen inference.'))
                optional_visualization = True
    for start, end, replacement in sorted(edits, reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    result = raw.decode('utf-8')
    compile(result, relative_path, 'exec')  # never write a malformed patch
    return result, dict(annotations=annotation_count,
                        optional_visualization=optional_visualization)
