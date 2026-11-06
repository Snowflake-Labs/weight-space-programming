"""CPU harness protocol/schema checks; no repair candidates are evaluated."""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

from weight_space.car_policy import SHAPES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('car_oneshot_tests', ROOT / 'scripts/car_oneshot.py')
car = importlib.util.module_from_spec(spec)
spec.loader.exec_module(car)


def test_preregistered_protocol():
    assert car.PROTOCOL['episode_ids'] == list(range(1024))
    assert car.PROTOCOL['seed'] == 20261105
    assert car.PROTOCOL['max_steps'] == 256
    assert car.PROTOCOL['frame_skip'] == 2
    assert not car.PROTOCOL['feedback_to_programmer']
    assert not car.PROTOCOL['paid_model_call_authorized']
    assert not car.PROTOCOL['context_ready']


def test_protocol_change_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(car, 'STATE', tmp_path)
    changed = dict(car.PROTOCOL, seed=0)
    (tmp_path / 'protocol.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='protocol changed'):
        car.load_protocol()


@pytest.mark.parametrize('violation', ['shape', 'dtype', 'nan', 'missing', 'object'])
def test_archive_schema_rejected(tmp_path, violation):
    tensors = {name: np.ones(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    if violation == 'shape':
        tensors['embed.b'] = np.zeros((127,), dtype=np.float32)
    elif violation == 'dtype':
        tensors['embed.b'] = tensors['embed.b'].astype(np.float64)
    elif violation == 'nan':
        tensors['embed.b'][0] = np.nan
    elif violation == 'missing':
        tensors.pop('embed.b')
    else:
        tensors['embed.b'] = np.array([object()], dtype=object)
    buffer = io.BytesIO()
    np.savez(buffer, **tensors)
    path = tmp_path / 'test.npz'
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError):
        car.load_weights(path)


def test_pinned_cpu_scene_and_metadata():
    pytest.importorskip('jax')
    pytest.importorskip('distrax')
    if not car.SCENE.exists():
        pytest.skip('Pinned external evaluator not bundled')
    _, metadata = car.build_evaluator()
    assert metadata['jax'] == '0.4.38'
    assert metadata['initial_observation_shape'] == [679]
    assert metadata['static']['frame_skip'] == 2
    assert metadata['initial_motor_probabilities'][0] > 0.999
    np.testing.assert_allclose(metadata['initial_body_xy'], [2.8, 0.725], atol=0.002)
