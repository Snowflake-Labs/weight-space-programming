"""Isolated published CPU control, then exactly one clean guided Astra request."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import car_astra_programmer as astra
from weight_space.car_policy import digest, exclusive_write, freeze_reply, reconstruct

ROOT = astra.ROOT
BASE = astra.BASE
CONTROL = ROOT / 'artifacts/car-published-control'
GUIDED = ROOT / 'artifacts/car-astra-guided'
INSTRUCTION = ('Consider whether the repair requires different actions in different physical states rather than a constant action preference. '
               'Define the conditions using the available observations, map those conditions through the network’s activations, '
               'and check the signs and thresholds of your proposed numerical edits. You must determine the weights yourself.')
ADDITION = ('ADDITIONAL INSTRUCTION\n' + INSTRUCTION + '\n\n').encode('utf-8')
SOURCE = 'repos/RikiyaT/lm_understand-logs/contents/exp4_fix_car/runs/oneshot_repair/reply/answer.md?ref=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
SOURCE_URL = 'https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car/runs/oneshot_repair/reply/answer.md'


def clean_request():
    original = (BASE / 'source/prompt.txt').read_bytes()
    tensors, _ = reconstruct(original)
    text = ADDITION + original
    request = astra.payload(text.decode('utf-8'), astra.MAX_OUTPUT)
    if request['messages'][1]['content'].encode('utf-8') != text:
        raise ValueError('Guided prompt roundtrip mismatch')
    return original, tensors, text, request


def prepare(directory):
    directory.mkdir(mode=0o700)
    hashes = {}
    for name in ('protocol.json', 'manifest.json', 'rounded-original.npz', 'baseline-1024.json'):
        data = (BASE / name).read_bytes()
        exclusive_write(directory / name, data)
        hashes[name] = digest(data)
    return hashes


def evaluate(directory):
    import car_oneshot as car
    import numpy as np
    car.STATE = directory
    car.evaluate_policy(1024, 32, True)
    edited = json.loads((directory / 'edited-1024.json').read_bytes())
    baseline = json.loads((directory / 'baseline-1024.json').read_bytes())
    freeze = json.loads((directory / 'freeze.json').read_bytes())
    original = car.load_weights(directory / 'rounded-original.npz')
    weights = car.load_weights(directory / 'edited.npz')
    squared = sum(float(np.sum((weights[name].astype(np.float64) - original[name].astype(np.float64)) ** 2)) for name in original)
    fields = ('episodes', 'success', 'red', 'timeout', 'invalid', 'mean_return', 'mean_length', 'wall_seconds')
    summary = {'policy': directory.name, 'source_authorship': 'published' if directory == CONTROL else 'fresh_guided_astra',
               'baseline': {key: baseline[key] for key in fields}, 'edited': {key: edited[key] for key in fields},
               'changed_coordinates': freeze['changed_coordinates'], 'total_coordinates': 171664,
               'applied_statements': len(freeze['applied']), 'skipped_statements': len(freeze['skipped']),
               'weight_delta_l2': squared ** 0.5, 'paired': json.loads((directory / 'paired-1024.json').read_bytes()),
               'paper_bit_exact': False, 'grader_changed': False,
               'artifact_sha256': {name: digest((directory / name).read_bytes()) for name in ('reply.txt', 'freeze.json', 'edited.npz', 'edited-1024.json', 'paired-1024.json')}}
    if directory == CONTROL:
        summary.update(reported_reference_coordinates=6073, reported_reference_statements=51,
                       coordinate_count_matches_report=freeze['changed_coordinates'] == 6073,
                       statement_count_matches_report=len(freeze['applied']) == 51,
                       positive_control_validated=edited['success'] == 1024)
    astra.save(directory / 'execution-summary.json', summary, exclusive=True)
    print(json.dumps(summary), flush=True)


def evaluate_child(directory):
    return subprocess.run([str(ROOT / '.venv-car/bin/python'), str(Path(__file__).resolve()),
                           '--evaluate', str(directory)], check=False,
                          env=dict(os.environ, JAX_PLATFORMS='cpu', WANDB_MODE='disabled')).returncode


def published_control():
    hashes = prepare(CONTROL)
    record = {'source_authorship': 'published', 'source_url': SOURCE_URL, 'source_commit': '4d09ba6d6a7e891dd8063974dfe81b3d78e3494f',
              'model_calls': 0, 'started_at': astra.now(), 'baseline_source_hashes': hashes, 'grader_changed': False}
    try:
        response = subprocess.run(['gh', 'api', SOURCE], capture_output=True, check=True, timeout=60)
        body = json.loads(response.stdout)
        if body['encoding'] != 'base64' or body['path'] != 'exp4_fix_car/runs/oneshot_repair/reply/answer.md':
            raise ValueError('Unexpected pinned GitHub source')
        answer = base64.b64decode(body['content'])
        exclusive_write(CONTROL / 'answer.md', answer)
        tensors, _ = reconstruct((BASE / 'source/prompt.txt').read_bytes())
        record.update(source_blob_sha=body['sha'], source_answer_sha256=digest(answer),
                      freeze=freeze_reply(CONTROL, tensors, answer, json.loads((CONTROL / 'protocol.json').read_bytes())))
        astra.save(CONTROL / 'source-audit.json', record, exclusive=True)
        if record['freeze']['status'] == 'frozen':
            record['evaluation_exit_code'] = evaluate_child(CONTROL)
        else:
            record['evaluation_exit_code'] = None
        record['status'] = 'evaluated' if (CONTROL / 'execution-summary.json').exists() else 'control_failed'
    except Exception as error:
        record.update(status='control_failed', error_type=type(error).__name__, error=str(error))
    record['finished_at'] = astra.now()
    astra.save(CONTROL / 'control-receipt.json', record, exclusive=True)
    print(json.dumps({'control_status': record['status'], 'guided_follows_regardless': True}), flush=True)


def guided():
    import requests
    from requests.adapters import HTTPAdapter
    logging.disable(logging.CRITICAL)
    token = astra.credential()
    astra.endpoint()
    original, tensors, text, request = clean_request()
    hashes = prepare(GUIDED)
    astra.OUTPUT = GUIDED
    record = {'experiment': 'guided-state-dependent', 'status': 'prepared', 'started_at': astra.now(),
              'base_prompt_sha256': digest(original), 'base_prompt_bytes': len(original),
              'addition_sha256': digest(ADDITION), 'addition_bytes': len(ADDITION), 'exact_instruction': INSTRUCTION,
              'addition_placement': 'prefix before original task; entire original is unchanged suffix',
              'guided_prompt_sha256': digest(text), 'guided_prompt_bytes': len(text), 'baseline_source_hashes': hashes,
              'requested_model': astra.MODEL, 'main_calls': 0, 'no_feedback': True,
              'no_author_answer_in_payload': True, 'no_prior_model_reply_in_payload': True,
              'actual_bill_usd': None, 'grader_changed': False}
    astra.save(GUIDED / 'run-receipt.json', record, exclusive=True)
    try:
        record.update(status='main_dispatched', main_calls=1)
        astra.save(GUIDED / 'run-receipt.json', record)
        print(json.dumps({'status': record['status'], 'guided_prompt_bytes': len(text)}), flush=True)
        with requests.Session() as session:
            session.trust_env = False
            session.verify = True
            session.mount('https://', HTTPAdapter(max_retries=0))
            reply, errors = astra.post(session, token, request, 'main')
        if errors:
            record.update(status='main_failed_consumed', errors=errors)
        else:
            record['freeze'] = freeze_reply(GUIDED, tensors, reply, json.loads((GUIDED / 'protocol.json').read_bytes()))
            record['status'] = record['freeze']['status']
    except Exception as error:
        record.update(status='blocked', error_type=type(error).__name__, errors=['Failure details suppressed for credential safety; no retry'])
    finally:
        token = None
        record['baseline_sources_unchanged'] = all(digest((BASE / name).read_bytes()) == value for name, value in hashes.items())
        record['finished_at'] = astra.now()
        astra.save(GUIDED / 'run-receipt.json', record)
        print(json.dumps({'status': record['status'], 'main_calls': record['main_calls']}), flush=True)
    if record['status'] == 'frozen':
        evaluate_child(GUIDED)


def audit():
    """Verify persisted payload and all comparators without inference or rollouts."""
    import numpy as np
    from weight_space.car_policy import apply_reply, tensor_digest
    original, tensors, text, expected = clean_request()
    request = json.loads((GUIDED / 'main-request.json').read_bytes())
    if request != expected or request['messages'][1]['content'].encode() != text:
        raise ValueError('Persisted guided request differs from approved clean payload')
    receipt = json.loads((GUIDED / 'run-receipt.json').read_bytes())
    main = json.loads((GUIDED / 'main-receipt.json').read_bytes())
    if main['status'] != 'validated' or main['dispatch_count'] != 1 or receipt['main_calls'] != 1:
        raise ValueError('Not a validated singleton guided request')
    summaries = {}
    for directory in (CONTROL, GUIDED):
        freeze = json.loads((directory / 'freeze.json').read_bytes())
        raw = (directory / 'reply.txt').read_bytes()
        weights, parsed = apply_reply(tensors, raw.decode())
        with np.load(directory / 'edited.npz', allow_pickle=False) as archive:
            stored = {name: archive[name] for name in archive.files}
        if (digest(raw) != freeze['reply_sha256'] or tensor_digest(weights) != tensor_digest(stored)
                or digest((directory / 'edited.npz').read_bytes()) != freeze['edited_archive_sha256']):
            raise ValueError('Frozen policy reconstruction mismatch')
        summaries[directory.name] = json.loads((directory / 'execution-summary.json').read_bytes())
        if summaries[directory.name]['skipped_statements'] != len(parsed['skipped']):
            raise ValueError('Parser audit mismatch')
    unguided = json.loads((ROOT / 'artifacts/car-astra-oneshot/execution-summary.json').read_bytes())
    result = {'control': summaries[CONTROL.name], 'guided': summaries[GUIDED.name],
              'historical_unguided_astra': unguided['astra'],
              'requested_model': main['requested_model'], 'returned_model': main['returned_model'],
              'finish_reason': main['finish_reason'], 'usage': main['usage'], 'model_wall_seconds': main['wall_seconds'],
              'exact_saved_payload_verified': True, 'original_prompt_sha256': digest(original),
              'guided_prompt_sha256': digest(text), 'request_sha256': digest((GUIDED / 'main-request.json').read_bytes()),
              'actual_bill_usd': None, 'pricing_scenario_usd_not_invoice': main['usage']['prompt_tokens'] * 100 / 1e6 + main['usage']['completion_tokens'] * 200 / 1e6,
              'no_feedback_or_second_repair': True, 'grader_changed': False, 'paper_bit_exact': False,
              'verification': json.loads((GUIDED / 'verification.json').read_bytes()),
              'source_sha256': {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in (Path(__file__), ROOT / 'scripts/car_astra_programmer.py', ROOT / 'scripts/car_oneshot.py', ROOT / 'src/weight_space/car_policy.py')}}
    astra.save(GUIDED / 'comparison-audit.json', result, exclusive=True)
    print(json.dumps(result, indent=2), flush=True)


def run():
    os.umask(0o077)
    if CONTROL.exists() or GUIDED.exists():
        raise ValueError('Experiment output exists; refuse overwrite or repeated model request')
    clean_request()
    published_control()
    # Deliberately do not gate this user-authorized test on the control result.
    guided()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-authorized-sequence', action='store_true')
    group.add_argument('--audit', action='store_true')
    group.add_argument('--evaluate', type=Path, choices=[CONTROL, GUIDED])
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.evaluate:
        evaluate(args.evaluate)
    else:
        run()
