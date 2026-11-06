"""Numeric-only car policy reconstruction and one-response freezing; no inference API."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

PROMPT_SHA256 = '2ed66051f4fa554d43d0163eb48e4256dc80aef78d875a218826beefeb7d0559'
CHECKPOINT_SHA256 = 'b0cdbccd5b4751d5a6dc4cb0deb18a036e6d08d71ad34802cea44dd26c6f4d11'
KINETIX_COMMIT = '80ee9c292837c825cafaf9889323fdcc212efd1b'
SHAPES = {'embed.W': (679, 128), 'embed.b': (128,)}
for layer in range(5):
    SHAPES.update({f'actor{layer}.W': (128, 128), f'actor{layer}.b': (128,)})
SHAPES.update({'actor_head.W': (128, 16), 'actor_head.b': (16,)})
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
HEADER = re.compile(r'^(embed|actor[0-4]|actor_head)\.([Wb]) \((\d+)(?: x (\d+))?\), scale (1e-\d+):.*$', re.M)
STATEMENT = re.compile(r'^(embed|actor[0-4]|actor_head)\.([Wb])\[([^\]]+)\]\s*(=|\*=|\+=|-=)\s*(.+)$', re.S)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(tensors):
        hasher.update(name.encode('ascii'))
        hasher.update(json.dumps(list(tensors[name].shape)).encode('ascii'))
        hasher.update(np.asarray(tensors[name], dtype='<f4').tobytes(order='C'))
    return hasher.hexdigest()


def reconstruct(prompt: bytes) -> tuple[dict, dict]:
    if len(prompt) != 577774 or digest(prompt) != PROMPT_SHA256:
        raise ValueError('Original prompt bytes/hash mismatch')
    text = prompt.decode('utf-8')
    headers = list(HEADER.finditer(text))
    tensors, scales = {}, {}
    for index, match in enumerate(headers):
        name = match[1] + '.' + match[2]
        shape = (int(match[3]),) if match[4] is None else (int(match[3]), int(match[4]))
        if name in tensors or SHAPES.get(name) != shape:
            raise ValueError('Unexpected or duplicate tensor header')
        body = text[match.end():headers[index + 1].start() if index + 1 < len(headers) else len(text)].strip()
        if len(shape) == 2:
            rows = body.splitlines()
            if len(rows) != shape[0]:
                raise ValueError(f'Incorrect row count: {name}')
            values = []
            for row_index, row in enumerate(rows):
                prefix, integers = row.split(':', 1)
                if prefix != str(row_index):
                    raise ValueError('Row indices must be consecutive')
                tokens = integers.split()
                if len(tokens) != shape[1]:
                    raise ValueError('Incorrect column count')
                values.extend(tokens)
        else:
            values = body.split()
        if len(values) != int(np.prod(shape)) or any(not re.fullmatch(r'-?\d+', value) for value in values):
            raise ValueError('Invalid integer tensor data')
        scales[name] = float(match[5])
        tensors[name] = (np.array(values, dtype=np.float64) * scales[name]).astype(np.float32).reshape(shape)
    if set(tensors) != set(SHAPES):
        raise ValueError('Incomplete actor')
    return tensors, scales


def verify_checkpoint(checkpoint: bytes, tensors: dict, scales: dict) -> dict:
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Original checkpoint hash mismatch')
    paths = {'embed': 'params/Dense_0', 'actor_head': 'params/GeneralActorCriticRNN_0/Dense_0'}
    paths.update({f'actor{index}': f'params/GeneralActorCriticRNN_0/DenseAndActivation_{index}/Dense_0' for index in range(5)})
    evidence = {}
    with np.load(io.BytesIO(checkpoint), allow_pickle=False) as archive:
        for name, shown in tensors.items():
            alias, suffix = name.split('.')
            original = archive[paths[alias] + ('/kernel' if suffix == 'W' else '/bias')]
            if original.shape != shown.shape or original.dtype != np.float32 or not np.isfinite(original).all():
                raise ValueError('Original checkpoint schema mismatch')
            scale = scales[name]
            # Compare integers, not a second float32 multiplication's last-bit rounding.
            matches = np.array_equal(np.rint(original.astype(np.float64) / scale), np.rint(shown.astype(np.float64) / scale))
            evidence[name] = {'rounding_matches': matches, 'max_abs_rounding_error': float(np.max(np.abs(original - shown)))}
            if not matches:
                raise ValueError(f'Checkpoint rounding mismatch: {name}')
    return evidence


def _indices(text: str, shape: tuple) -> tuple:
    axes = text.split(',')
    if len(axes) != len(shape):
        raise ValueError('Index rank mismatch')
    result = []
    for axis, size in zip(axes, shape):
        axis = axis.strip()
        if re.fullmatch(r'\d+', axis):
            value = int(axis)
            if value >= size:
                raise ValueError('Index out of range')
            result.append(value)
        elif re.fullmatch(r'\d*:\d*', axis):
            start, stop = axis.split(':')
            start, stop = int(start or 0), int(stop or size)
            if not 0 <= start < stop <= size:
                raise ValueError('Range out of bounds or empty')
            result.append(slice(start, stop))
        else:
            raise ValueError('Only nonnegative indices and half-open ranges allowed')
    return tuple(result)


def _values(text: str):
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        parts = text[1:-1].split(',')
        if not parts or any(not re.fullmatch(NUMBER, part.strip()) for part in parts):
            raise ValueError('Flat decimal list only; no empty values or trailing comma')
        return np.array([_values(part) for part in parts], dtype=np.float32)
    if not re.fullmatch(NUMBER, text):
        raise ValueError('Plain decimal numbers only')
    value = float(text)
    if not np.isfinite(value) or abs(value) > 1e6:
        raise ValueError('Numeric literal exceeds finite bound')
    return np.float32(value)


def apply_reply(tensors: dict, reply: str) -> tuple[dict, dict]:
    blocks = re.findall(r'^```edits[^\S\n]*\n(.*?)^```[^\S\n]*$', reply, re.M | re.S)
    if not blocks:
        raise ValueError('No complete edits block')
    result = {name: value.copy() for name, value in tensors.items()}
    statements, pending = [], ''
    for line in blocks[-1].splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if pending and (STATEMENT.fullmatch(line) or line.startswith('ZERO ')):
            statements.append(pending)
            pending = ''
        if pending:
            pending += ' ' + line
        else:
            pending = line
        if pending.count('[') <= pending.count(']'):
            statements.append(pending)
            pending = ''
    if pending:
        statements.append(pending)
    applied, skipped = [], []
    for statement in statements:
        try:
            zero = re.fullmatch(r'ZERO\s+(embed|actor[0-4]|actor_head)\.([Wb])', statement)
            if zero:
                name = zero[1] + '.' + zero[2]
                result[name].fill(0)
            else:
                match = STATEMENT.fullmatch(statement)
                if not match:
                    raise ValueError('Unrecognized numeric edit')
                name = match[1] + '.' + match[2]
                indices = _indices(match[3], result[name].shape)
                values = _values(match[5])
                old = np.asarray(result[name][indices])
                if isinstance(values, np.ndarray) and values.shape != old.shape:
                    raise ValueError('Explicit list must match the selected one-dimensional range')
                with np.errstate(over='raise', invalid='raise'):
                    if match[4] == '=':
                        new = np.full(old.shape, values, dtype=np.float32) if np.ndim(values) == 0 else values
                    elif match[4] == '*=':
                        new = old * values
                    elif match[4] == '+=':
                        new = old + values
                    else:
                        new = old - values
                if not np.isfinite(new).all():
                    raise ValueError('Edit produced non-finite weights')
                result[name][indices] = new
            applied.append(statement)
        except (ValueError, FloatingPointError) as error:
            skipped.append({'statement': statement, 'reason': str(error)})
    changed = sum(int(np.count_nonzero(result[name] != tensors[name])) for name in tensors)
    return result, {'blocks_found': len(blocks), 'applied': applied, 'skipped': skipped,
                    'changed_coordinates': changed, 'total_coordinates': sum(value.size for value in tensors.values()),
                    'nonidentity': bool(changed), 'tensor_sha256': tensor_digest(result),
                    'parser_policy': 'last block; sequential edits; unreadable statements skipped and audited as original prompt specifies'}


def exclusive_write(path: Path, data: bytes):
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def freeze_reply(directory: Path, tensors: dict, raw_reply: bytes, protocol: dict) -> dict:
    """First reply consumes the freeze even if invalid; never select or repair another."""
    exclusive_write(directory / 'reply.txt', raw_reply)
    record = {'reply_sha256': digest(raw_reply), 'original_tensor_sha256': tensor_digest(tensors),
              'protocol_sha256': digest(json.dumps(protocol, sort_keys=True).encode()), 'status': 'invalid'}
    try:
        edited, audit = apply_reply(tensors, raw_reply.decode('utf-8'))
        record.update(audit)
        if not audit['nonidentity']:
            raise ValueError('Identity is not an accepted repair')
        buffer = io.BytesIO()
        np.savez(buffer, **edited)
        exclusive_write(directory / 'edited.npz', buffer.getvalue())
        record.update(status='frozen', edited_archive_sha256=digest(buffer.getvalue()))
    except (ValueError, UnicodeError) as error:
        record['error'] = str(error)
    exclusive_write(directory / 'freeze.json', (json.dumps(record, indent=2) + '\n').encode())
    return record
