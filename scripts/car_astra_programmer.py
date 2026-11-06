"""One bounded Astra REST repair, frozen before paired local CPU evaluation."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from weight_space.car_policy import digest, reconstruct, freeze_reply, exclusive_write, tensor_digest

BASE = ROOT / 'artifacts/car-oneshot'
OUTPUT = ROOT / 'artifacts/car-astra-oneshot'
MODEL = 'openai-gpt-6-astra'
COMPLETIONS_PATH = '/api/v2/cortex/v1/chat/completions'
SYSTEM = "Answer the user's request."
MAX_OUTPUT = 65536
DOC_URL = 'https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value, exclusive=False):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if exclusive:
        exclusive_write(path, data)
    else:
        with path.open('wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def endpoint():
    """Cortex REST completions URL from SNOWFLAKE_ACCOUNT_URL, e.g. https://<account>.snowflakecomputing.com."""
    base = os.environ.get('SNOWFLAKE_ACCOUNT_URL', '').rstrip('/')
    if not base.startswith('https://') or not base.endswith('.snowflakecomputing.com'):
        raise ValueError('Set SNOWFLAKE_ACCOUNT_URL to https://<account>.snowflakecomputing.com')
    return base + COMPLETIONS_PATH


def credential():
    """Programmatic access token from SNOWFLAKE_PAT; never persisted."""
    token = os.environ.get('SNOWFLAKE_PAT', '')
    if not token or any(char in token for char in '\r\n"'):
        raise ValueError('Set SNOWFLAKE_PAT to a programmatic access token')
    return token


def payload(text, limit):
    return {'model': MODEL, 'messages': [{'role': 'system', 'content': SYSTEM},
                                         {'role': 'user', 'content': text}],
            'reasoning_effort': 'high', 'max_completion_tokens': limit, 'stream': False}


def inspect_response(body, status):
    errors = []
    if status != 200 or body.get('error') or body.get('code') not in (None, 0, '0'):
        errors.append('HTTP/API failure (including session-expiry errors in HTTP200)')
    if body.get('model') != MODEL:
        errors.append('Returned model does not exactly match requested model; no alias fallback')
    choices = body.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        errors.append('Expected exactly one choice')
        return b'', errors
    message = choices[0].get('message', {})
    if message.get('role') != 'assistant' or message.get('tool_calls') or message.get('function_call') or message.get('refusal'):
        errors.append('Unexpected role, tool call, or refusal')
    content = message.get('content')
    if not isinstance(content, str) or not content:
        errors.append('No completion text')
        content = ''
    if choices[0].get('finish_reason') != 'stop':
        errors.append('Completion did not finish normally; first response retained without retry')
    return content.encode(), errors


def post(session, credential, request, name):
    url = endpoint()
    save(OUTPUT / (name + '-request.json'), request, exclusive=True)
    committed = {'dispatched_at': now(), 'dispatch_count': 1, 'requested_model': MODEL,
                 'request_sha256': digest((OUTPUT / (name + '-request.json')).read_bytes()),
                 'max_completion_tokens': request['max_completion_tokens'], 'reasoning_effort': 'high',
                 'tls_verify': True, 'redirects': False, 'retries': 0, 'timeout_seconds': 1200,
                 'status': 'dispatch_committed', 'auth_type': 'PAT'}
    receipt_path = OUTPUT / (name + '-receipt.json')
    save(receipt_path, committed, exclusive=True)
    start = time.monotonic()
    response = None
    try:
        response = session.post(url, headers={'Authorization': f'Bearer {credential}', 'Content-Type': 'application/json'},
                                json=request, timeout=(30, 1200), allow_redirects=False, verify=True)
        raw = response.content.replace(credential.encode(), b'[REDACTED]')
        exclusive_write(OUTPUT / (name + '-response-body.json'), raw)
        # Record response headers only; never persist request Authorization or cookies.
        headers = {key: value.replace(credential, '[REDACTED]') for key, value in response.headers.items()
                   if key.lower() not in ('set-cookie', 'authorization', 'proxy-authorization')}
        save(OUTPUT / (name + '-response-headers.json'), headers, exclusive=True)
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError('Response is not an object')
        text, errors = inspect_response(body, response.status_code)
        if name == 'main' and text:
            exclusive_write(OUTPUT / 'first-response.txt', text)
        committed.update(http_status=response.status_code, returned_model=body.get('model'),
                         finish_reason=body.get('choices', [{}])[0].get('finish_reason') if body.get('choices') else None,
                         usage=body.get('usage'), response_sha256=digest(raw), errors=errors,
                         response_error={key: body[key] for key in ('code', 'message', 'error') if key in body},
                         status='validated' if not errors else 'failed_consumed')
        return text, errors
    except Exception as error:
        committed.update(status='failed_consumed', error_type=type(error).__name__,
                         errors=['Transport or decoding failure; details suppressed for credential safety; no retry'])
        return b'', committed['errors']
    finally:
        if response is not None:
            response.close()
        committed.update(finished_at=now(), wall_seconds=time.monotonic() - start)
        save(receipt_path, committed)


def prepare():
    if OUTPUT.exists():
        raise ValueError('Astra output already exists; no repeat or overwrite')
    prompt = (BASE / 'source/prompt.txt').read_bytes()
    tensors, _ = reconstruct(prompt)
    if sum(value.size for value in tensors.values()) != 171664:
        raise ValueError('Incomplete rounded actor')
    OUTPUT.mkdir(mode=0o700)
    baseline_hashes = {}
    for name in ('protocol.json', 'manifest.json', 'rounded-original.npz', 'baseline-1024.json'):
        data = (BASE / name).read_bytes()
        exclusive_write(OUTPUT / name, data)
        baseline_hashes[name] = digest(data)
    return prompt, tensors, baseline_hashes


def run():
    import requests
    from requests.adapters import HTTPAdapter
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    token = credential()
    endpoint()
    prompt, tensors, baseline_hashes = prepare()
    record = {'started_at': now(), 'status': 'prepared', 'requested_model': MODEL,
              'prompt_bytes': len(prompt), 'prompt_sha256': digest(prompt),
              'actor_coordinates': 171664, 'baseline_source_hashes': baseline_hashes,
              'neutral_system': SYSTEM, 'no_tools': True, 'no_feedback': True, 'no_retries': True,
              'main_calls': 0, 'actual_bill_usd': None, 'hard_billing_cap': False,
              'documentation': DOC_URL, 'paper_bit_exact': False}
    save(OUTPUT / 'run-receipt.json', record, exclusive=True)
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.verify = True
            session.mount('https://', HTTPAdapter(max_retries=0))
            _, errors = post(session, token, payload('Reply only hello.', 1024), 'preflight')
            if errors:
                record.update(status='preflight_failed_main_not_dispatched', errors=errors)
                return
            record.update(preflight_verified=True, status='main_dispatched', main_calls=1)
            save(OUTPUT / 'run-receipt.json', record)
            request = payload(prompt.decode(), MAX_OUTPUT)
            if request['messages'][1]['content'].encode() != prompt:
                raise ValueError('Prompt roundtrip mismatch')
            text, errors = post(session, token, request, 'main')
            if errors:
                record.update(status='main_failed_consumed', errors=errors)
                return
            protocol = json.loads((OUTPUT / 'protocol.json').read_bytes())
            record['freeze'] = freeze_reply(OUTPUT, tensors, text, protocol)
            record['status'] = record['freeze']['status']
    except Exception as error:
        record.update(status='blocked', error_type=type(error).__name__,
                      errors=['Details suppressed for credential safety; no automatic retry'])
    finally:
        token = None
        record['baseline_sources_unchanged'] = all(digest((BASE / name).read_bytes()) == value for name, value in baseline_hashes.items())
        record['finished_at'] = now()
        save(OUTPUT / 'run-receipt.json', record)
        print(json.dumps({'status': record['status'], 'main_calls': record['main_calls'],
                          'errors': record.get('errors'), 'receipt': str(OUTPUT / 'run-receipt.json')}), flush=True)
    if record['status'] == 'frozen':
        subprocess.run([str(ROOT / '.venv-car/bin/python'), str(Path(__file__).resolve()), '--evaluate'], check=True,
                       env=dict(os.environ, JAX_PLATFORMS='cpu', WANDB_MODE='disabled'))


def evaluate():
    import car_oneshot as car
    import numpy as np
    os.umask(0o077)
    receipt = json.loads((OUTPUT / 'run-receipt.json').read_bytes())
    if receipt['status'] != 'frozen' or receipt['main_calls'] != 1:
        raise ValueError('No valid singleton frozen Astra response')
    car.STATE = OUTPUT
    car.evaluate_policy(1024, 32, True)
    baseline = json.loads((OUTPUT / 'baseline-1024.json').read_bytes())
    edited = json.loads((OUTPUT / 'edited-1024.json').read_bytes())
    freeze = json.loads((OUTPUT / 'freeze.json').read_bytes())
    main = json.loads((OUTPUT / 'main-receipt.json').read_bytes())
    original_weights = car.load_weights(OUTPUT / 'rounded-original.npz')
    edited_weights = car.load_weights(OUTPUT / 'edited.npz')
    squared = sum(float(np.sum((edited_weights[name].astype(np.float64) - original_weights[name].astype(np.float64)) ** 2)) for name in original_weights)
    usage = main['usage']
    summary = {'requested_model': MODEL, 'returned_model': main['returned_model'],
               'finish_reason': main['finish_reason'], 'usage': usage, 'model_wall_seconds': main['wall_seconds'],
               'baseline': {key: baseline[key] for key in ('episodes', 'success', 'red', 'timeout', 'invalid', 'mean_return', 'mean_length')},
               'astra': {key: edited[key] for key in ('episodes', 'success', 'red', 'timeout', 'invalid', 'mean_return', 'mean_length', 'wall_seconds')},
               'changed_coordinates': freeze['changed_coordinates'], 'changed_fraction': freeze['changed_coordinates'] / 171664,
               'weight_delta_l2': squared ** 0.5, 'applied_statements': len(freeze['applied']), 'skipped_statements': len(freeze['skipped']),
               'paired': json.loads((OUTPUT / 'paired-1024.json').read_bytes()),
               'pricing_scenario_usd_not_invoice': usage['prompt_tokens'] * 100 / 1e6 + usage['completion_tokens'] * 200 / 1e6,
               'paper_bit_exact': False, 'no_feedback_or_second_repair': True,
               'artifact_sha256': {name: digest((OUTPUT / name).read_bytes()) for name in ('reply.txt', 'freeze.json', 'edited.npz', 'edited-1024.json', 'paired-1024.json', 'main-response-body.json', 'main-request.json')},
               'baseline_sources_unchanged': all(digest((BASE / name).read_bytes()) == value for name, value in receipt['baseline_source_hashes'].items())}
    save(OUTPUT / 'execution-summary.json', summary, exclusive=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-authorized-one-shot', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        run()
