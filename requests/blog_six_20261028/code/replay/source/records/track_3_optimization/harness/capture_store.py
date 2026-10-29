import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


class CaptureWriter:
    """Write one rank's immutable tensor files and durable hash receipts.

    Reading and replay belong to the offline readers, not this writer.
    """

    def __init__(self, root, rank):
        self.root = Path(root) / f'rank{rank}'
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, kind, step, payload):
        if kind not in {'gradient', 'boundary', 'anchor', 'action', 'local_muon'} or step < 0:
            raise ValueError('invalid capture identity')
        target = self.root / f'{kind}_{step:06d}.pt'
        descriptor, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.root)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                torch.save(cpu_copy(payload), stream)
                stream.flush()
                os.fsync(stream.fileno())
            # link publishes atomically and refuses an existing target.
            os.link(temporary, target)
            receipt = {'kind': kind, 'step': step, 'file': target.name,
                       'sha256': file_sha(target), 'bytes': target.stat().st_size}
            with (self.root / 'manifest.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(receipt, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            return receipt
        finally:
            os.unlink(temporary)
