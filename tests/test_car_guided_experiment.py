"""Offline guards for the authorized isolated control/guided sequence."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import car_astra_programmer as astra
import car_guided_experiment as experiment


def test_exact_clean_request_and_no_source_dependency(monkeypatch):
    expected = ('Consider whether the repair requires different actions in different physical states rather than a constant action preference. '
                'Define the conditions using the available observations, map those conditions through the network’s activations, '
                'and check the signs and thresholds of your proposed numerical edits. You must determine the weights yourself.')
    original, tensors, text, request = experiment.clean_request()
    assert experiment.INSTRUCTION == expected
    assert text == ('ADDITIONAL INSTRUCTION\n' + expected + '\n\n').encode() + original
    assert len(original) == 577774
    assert astra.digest(original) == '2ed66051f4fa554d43d0163eb48e4256dc80aef78d875a218826beefeb7d0559'
    assert sum(value.size for value in tensors.values()) == 171664
    assert request == astra.payload(text.decode(), 65536)
    assert len(request['messages']) == 2
    assert 'RikiyaT' not in request['messages'][1]['content']
    assert 'actor0.b[122:128] = [-200, -248' not in request['messages'][1]['content']
    assert 'positive_control' not in request['messages'][1]['content']
    # Control output cannot influence the payload: no control path is ever opened.
    monkeypatch.setattr(experiment, 'CONTROL', Path('/not-available/published-answer'))
    assert experiment.clean_request()[3] == request


def test_guided_requires_credential_before_any_output(monkeypatch, tmp_path):
    monkeypatch.delenv('SNOWFLAKE_PAT', raising=False)
    monkeypatch.setattr(experiment, 'GUIDED', tmp_path / 'guided')
    with pytest.raises(ValueError, match='SNOWFLAKE_PAT'):
        experiment.guided()
    assert not (tmp_path / 'guided').exists()


def test_failed_control_does_not_suppress_guided(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(experiment, 'CONTROL', tmp_path / 'control')
    monkeypatch.setattr(experiment, 'GUIDED', tmp_path / 'guided')
    monkeypatch.setattr(experiment, 'clean_request', lambda: None)
    monkeypatch.setattr(experiment, 'published_control', lambda: order.append('failed-control-recorded'))
    monkeypatch.setattr(experiment, 'guided', lambda: order.append('guided'))
    experiment.run()
    assert order == ['failed-control-recorded', 'guided']


def test_existing_output_prevents_repeat(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, 'CONTROL', tmp_path)
    with pytest.raises(ValueError, match='exists'):
        experiment.run()
