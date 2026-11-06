"""Synthetic grammar fixtures only: no car repair candidate or author solution."""
import io
from pathlib import Path

import numpy as np
import pytest

from weight_space.car_policy import SHAPES, apply_reply, freeze_reply, reconstruct, tensor_digest, verify_checkpoint


@pytest.fixture
def actor():
    return {name: np.ones(shape, dtype=np.float32) for name, shape in SHAPES.items()}


def reply(body):
    return 'Explanation\n```edits\n' + body + '\n```\nAfterward.'


def test_all_operations_in_order(actor):
    edited, audit = apply_reply(actor, reply('ZERO embed.b\nembed.b[0:3] = [1, 2, 3]\nembed.b[1] += 2\nembed.b[1] *= 0.5\nembed.b[2] -= 1'))
    np.testing.assert_array_equal(edited['embed.b'][:4], [1, 2, 2, 0])
    assert not audit['skipped']
    assert np.all(actor['embed.b'] == 1)


def test_matrix_ranges_and_last_block(actor):
    edited, audit = apply_reply(actor, reply('ZERO embed.W') + '\n' + reply('embed.W[:, 3] *= 0.5\nactor_head.W[7, 0:3] = [2,3,4]'))
    assert audit['blocks_found'] == 2
    assert np.all(edited['embed.W'][:, 3] == 0.5)
    assert np.all(edited['embed.W'][:, 2] == 1)
    np.testing.assert_array_equal(edited['actor_head.W'][7, :3], [2, 3, 4])


def test_multiline_list_and_comment(actor):
    edited, audit = apply_reply(actor, reply('embed.b[0:3] = [1, # comment\n 2,\n 3]\nembed.W[26,5] = -1.25'))
    assert len(audit['applied']) == 2
    np.testing.assert_array_equal(edited['embed.b'][:3], [1, 2, 3])
    assert edited['embed.W'][26, 5] == -1.25


@pytest.mark.parametrize('statement', [
    'embed.b[-1] = 0', 'embed.b[128] = 0', 'embed.b[0:129] = 0',
    'embed.b[2:1] = 0', 'embed.b[1:1] = 0', 'embed.b[::2] = 0',
    'embed.b[0,1] = 0', 'embed.W[0] = 0', 'critic.b[0] = 0',
    'embed.b[0] = nan', 'embed.b[0] = inf', 'embed.b[0] = 1e3',
    'embed.b[0] = 1000001', 'embed.b[0] = -1000001',
    'embed.b[0] = __import__("os").system("false")',
    'embed.b[0:3] = [1, 2]', 'embed.b[0:3] = [1,2,3,]',
    'embed.b[0] = [1]', 'embed.W[0:2,0:2] = [1,2]',
    'SET embed.b[0] 0', 'COPY actor0.W actor1.W', 'REPLACE embed.b 0',
])
def test_bad_statement_skipped_without_execution(actor, statement):
    edited, audit = apply_reply(actor, reply(statement + '\nembed.b[1] = 2'))
    assert len(audit['skipped']) == 1
    assert edited['embed.b'][0] == 1
    assert edited['embed.b'][1] == 2


def test_duplicate_edits_are_sequential_not_rejected(actor):
    edited, audit = apply_reply(actor, reply('embed.b[0] = 2\nembed.b[0] = 3'))
    assert edited['embed.b'][0] == 3
    assert len(audit['applied']) == 2


def test_float_overflow_is_skipped_atomically(actor):
    actor['embed.b'][0] = np.finfo(np.float32).max
    edited, audit = apply_reply(actor, reply('embed.b[0:3] *= 1000000'))
    assert len(audit['skipped']) == 1
    np.testing.assert_array_equal(edited['embed.b'], actor['embed.b'])


def test_no_block_or_unclosed_block(actor):
    for text in ['no edits', '```edits\nZERO embed.b']:
        with pytest.raises(ValueError):
            apply_reply(actor, text)


def test_unclosed_list_does_not_consume_next_statement(actor):
    edited, audit = apply_reply(actor, reply('embed.b[0:3] = [1,2\nembed.b[4] = 5'))
    assert len(audit['skipped']) == 1
    assert edited['embed.b'][4] == 5


def test_nested_lists_rejected(actor):
    edited, audit = apply_reply(actor, reply('embed.b[0:3] = [[1],[2],[3]]'))
    assert len(audit['skipped']) == 1
    assert not audit['nonidentity']


def test_no_changes_and_tensor_hash(actor):
    edited, audit = apply_reply(actor, reply('embed.b[:] = 1'))
    assert not audit['nonidentity']
    assert tensor_digest(edited) == tensor_digest(actor)


def test_freeze_first_reply_only_and_identity_rejected(actor, tmp_path):
    record = freeze_reply(tmp_path, actor, reply('embed.b[:] = 1').encode(), {'seed': 1})
    assert record['status'] == 'invalid'
    assert not (tmp_path / 'edited.npz').exists()
    with pytest.raises(FileExistsError):
        freeze_reply(tmp_path, actor, reply('ZERO embed.b').encode(), {'seed': 1})


def test_freeze_numeric_archive(actor, tmp_path):
    record = freeze_reply(tmp_path, actor, reply('embed.b[0] = 2').encode(), {'seed': 1})
    assert record['status'] == 'frozen'
    with np.load(io.BytesIO((tmp_path / 'edited.npz').read_bytes()), allow_pickle=False) as archive:
        assert archive['embed.b'][0] == 2
    assert record['changed_coordinates'] == 1


def test_prompt_hash_rejects_modified_input():
    with pytest.raises(ValueError):
        reconstruct(b'not original')


def test_actual_clean_prompt_and_checkpoint():
    source = Path(__file__).resolve().parents[1] / 'artifacts/car-oneshot/source'
    if not (source / 'prompt.txt').exists():
        pytest.skip('External original input not bundled with unit tests')
    tensors, scales = reconstruct((source / 'prompt.txt').read_bytes())
    assert len(tensors) == 14
    assert sum(value.size for value in tensors.values()) == 171664
    assert all(value.dtype == np.float32 for value in tensors.values())
    evidence = verify_checkpoint((source / 'original.npz').read_bytes(), tensors, scales)
    assert all(item['rounding_matches'] for item in evidence.values())
    with pytest.raises(ValueError):
        verify_checkpoint(b'not original', tensors, scales)
