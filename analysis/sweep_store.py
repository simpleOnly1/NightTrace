"""Atomic per-image checkpoints, strict resume contracts and shard validation."""
import csv
import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def checkpoint_path(out, sid):
    return Path(out)/'checkpoints'/(hashlib.sha256(sid.encode()).hexdigest()+'.json')


def initialize(out, contract, resume):
    path = Path(out)/'contract.json'
    if resume:
        if not path.is_file():
            raise ValueError('No resumable contract; old-format runs need a new output directory')
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != contract:
            changed = sorted(k for k in set(previous) | set(contract) if previous.get(k) != contract.get(k))
            raise ValueError('Resume contract mismatch: '+', '.join(changed))
    else:
        atomic_json(path, contract)


def read_checkpoint(out, sid):
    path = checkpoint_path(out, sid)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('id') != sid or not data.get('complete'):
        raise ValueError('Invalid checkpoint: '+str(path))
    return data


class CsvSink:
    """One pass, one open file. Never rewrite previously completed images."""
    def __init__(self, path, optional=()):
        self.stream = Path(path).open('w', newline='', encoding='utf-8')
        self.writer = None; self.optional = list(optional)

    def write(self, rows):
        for row in rows:
            if self.writer is None:
                columns = list(dict.fromkeys(list(row)+self.optional))
                self.writer = csv.DictWriter(self.stream, fieldnames=columns)
                self.writer.writeheader()
            self.writer.writerow(row)

    def close(self):
        self.stream.close()


def collect_shards(paths):
    """Require the complete, non-overlapping cohort before any pooled inference."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError('No run directories')
    contracts = [json.loads((p/'contract.json').read_text(encoding='utf-8')) for p in paths]
    reference = {k: v for k, v in contracts[0].items() if k != 'shard_index'}
    if len(paths) != reference['num_shards']:
        raise ValueError('Supply every shard, not a partial cohort')
    indices = [c['shard_index'] for c in contracts]
    if set(indices) != set(range(reference['num_shards'])) or len(set(indices)) != len(indices):
        raise ValueError('Duplicate or missing shard index')
    records = []; seen = set()
    for out, contract in zip(paths, contracts):
        if {k: v for k, v in contract.items() if k != 'shard_index'} != reference:
            raise ValueError('Incompatible shard contracts')
        for i, sample in enumerate(contract['samples']):
            if i % contract['num_shards'] != contract['shard_index']:
                continue
            sid = sample['id']; path = checkpoint_path(out, sid)
            if not path.is_file():
                raise ValueError('Incomplete run; missing checkpoint for '+sid)
            if sid in seen:
                raise ValueError('Duplicate sample across shards: '+sid)
            seen.add(sid); records.append((sid, path))
    return reference, sorted(records)
