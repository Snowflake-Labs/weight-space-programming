"""Offline-only Astra transport, configuration, and response tests."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('car_astra_tests', ROOT / 'scripts/car_astra_programmer.py')
astra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(astra)


def completion(**changes):
    body = {'model': astra.MODEL, 'choices': [{'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}]}
    body.update(changes)
    return body


def test_raw_payload():
    request = astra.payload('original\ntext', 65536)
    assert request == {'model': 'openai-gpt-6-astra', 'messages': [
        {'role': 'system', 'content': "Answer the user's request."},
        {'role': 'user', 'content': 'original\ntext'}],
        'reasoning_effort': 'high', 'max_completion_tokens': 65536, 'stream': False}


def test_exact_identity_and_normal_finish():
    assert astra.inspect_response(completion(), 200) == (b'hello', [])


@pytest.mark.parametrize('problem', ['alias', 'expired', 'tool', 'refusal', 'length', 'empty', 'multiple', 'http'])
def test_invalid_response_consumed(problem):
    body = completion()
    status = 200
    if problem == 'alias':
        body['model'] += '-dated'
    elif problem == 'expired':
        body['code'] = '390112'
    elif problem == 'tool':
        body['choices'][0]['message']['tool_calls'] = [{}]
    elif problem == 'refusal':
        body['choices'][0]['message']['refusal'] = 'no'
    elif problem == 'length':
        body['choices'][0]['finish_reason'] = 'length'
    elif problem == 'empty':
        body['choices'][0]['message']['content'] = None
    elif problem == 'multiple':
        body['choices'] *= 2
    else:
        status = 403
    assert astra.inspect_response(body, status)[1]


ACCOUNT_URL = 'https://example-account.snowflakecomputing.com'


@pytest.mark.parametrize('value', ['', 'http://example.snowflakecomputing.com', 'https://example.com'])
def test_endpoint_requires_snowflake_https_url(monkeypatch, value):
    monkeypatch.setenv('SNOWFLAKE_ACCOUNT_URL', value)
    with pytest.raises(ValueError, match='SNOWFLAKE_ACCOUNT_URL'):
        astra.endpoint()


def test_endpoint_and_credential_from_environment(monkeypatch):
    monkeypatch.setenv('SNOWFLAKE_ACCOUNT_URL', ACCOUNT_URL + '/')
    monkeypatch.setenv('SNOWFLAKE_PAT', 'token')
    assert astra.endpoint() == ACCOUNT_URL + '/api/v2/cortex/v1/chat/completions'
    assert astra.credential() == 'token'
    monkeypatch.setenv('SNOWFLAKE_PAT', 'bad\ntoken')
    with pytest.raises(ValueError, match='SNOWFLAKE_PAT'):
        astra.credential()


def test_post_no_retry_redirect_or_secret_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(astra, 'OUTPUT', tmp_path)
    monkeypatch.setenv('SNOWFLAKE_ACCOUNT_URL', ACCOUNT_URL)
    secret = 'sensitive-example-credential'
    class Response:
        status_code = 200
        content = json.dumps(completion()).encode()
        headers = {'x-request-id': 'request1', 'Set-Cookie': secret}
        def close(self):
            pass
    class Session:
        calls = 0
        def post(self, url, **options):
            self.calls += 1
            assert url == ACCOUNT_URL + '/api/v2/cortex/v1/chat/completions'
            assert options['allow_redirects'] is False
            assert options['verify'] is True
            assert options['timeout'] == (30, 1200)
            return Response()
    session = Session()
    assert astra.post(session, secret, astra.payload('hello', 1024), 'preflight') == (b'hello', [])
    assert session.calls == 1
    assert all(secret.encode() not in path.read_bytes() for path in tmp_path.iterdir())


def test_timeout_is_consumed_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(astra, 'OUTPUT', tmp_path)
    monkeypatch.setenv('SNOWFLAKE_ACCOUNT_URL', ACCOUNT_URL)
    class Session:
        calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError('never log secret')
    session = Session()
    assert astra.post(session, 'secret', astra.payload('hello', 1024), 'main')[1]
    assert session.calls == 1
    record = json.loads((tmp_path / 'main-receipt.json').read_bytes())
    assert record['status'] == 'failed_consumed'
    assert 'never log secret' not in json.dumps(record)
