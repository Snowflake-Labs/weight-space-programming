"""Local CPU preparation/evaluation only. Never calls a model or managed compute."""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from weight_space.car_policy import (CHECKPOINT_SHA256, KINETIX_COMMIT, PROMPT_SHA256,
                                    SHAPES, digest, exclusive_write, freeze_reply,
                                    reconstruct, tensor_digest, verify_checkpoint)

STATE = ROOT / 'artifacts/car-oneshot'
SOURCE = STATE / 'source'
KINETIX = SOURCE / 'FLAIROx-Kinetix-80ee9c2'
SCENE = KINETIX / 'kinetix/levels/l/h2_car_ramp.json'
PROTOCOL = {
    'schema_version': 1, 'seed': 20261105, 'episode_ids': list(range(1024)),
    'max_steps': 256, 'frame_skip': 2, 'dense_reward_scale': 0.2,
    'key_schedule': 'fold_in(PRNGKey(seed), episode_id); each step split into next, action, environment; official six-way action distribution',
    'comparison': 'same rounded prompt actor and same episode/step random keys for both policies',
    'one_shot': True, 'feedback_to_programmer': False, 'training_or_gradients': False,
    'prompt_sha256': PROMPT_SHA256, 'checkpoint_sha256': CHECKPOINT_SHA256,
    'kinetix_commit': KINETIX_COMMIT,
    'not_paper_bit_exact': 'Author evaluator, dependency lock and held-out PRNG schedule were not released; this is our preregistered reconstruction.',
    'termination': 'Pinned env first contact substep with red priority within substep; additionally terminate any nonfinite state. No autoreset.',
    'context_ready': False,
    'context_blocker': 'Installed CLI help has no verified 1M-context switch; full 355201-token input support and effective neutral-only system context remain unverified without inference.',
    'paid_model_call_authorized': False,
}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def load_protocol():
    value = json.loads((STATE / 'protocol.json').read_text())
    if value != PROTOCOL:
        raise ValueError('Preregistered protocol changed')
    return value


def prepare():
    tensors, scales = reconstruct((SOURCE / 'prompt.txt').read_bytes())
    evidence = verify_checkpoint((SOURCE / 'original.npz').read_bytes(), tensors, scales)
    protocol = dict(PROTOCOL)
    buffer = io.BytesIO()
    import numpy as np
    np.savez(buffer, **tensors)
    files = {'protocol.json': json_bytes(protocol), 'rounded-original.npz': buffer.getvalue(),
             'manifest.json': json_bytes({'prompt_bytes': 577774, 'prompt_sha256': PROMPT_SHA256,
                 'checkpoint_sha256': CHECKPOINT_SHA256, 'kinetix_commit': KINETIX_COMMIT,
                 'kinetix_archive_sha256': digest((SOURCE / 'kinetix.tar.gz').read_bytes()),
                 'scene_sha256': digest(SCENE.read_bytes()), 'rounded_tensor_sha256': tensor_digest(tensors),
                 'checkpoint_rounding': evidence, 'actor_coordinates': 171664,
                 'baseline_uses_full_precision_checkpoint': False,
                 'author_answer_or_repaired_weights_fetched': False,
                 'input_tokens_original_reported': 355201, 'output_tokens_original_reported': 10134,
                 'cpu_runtime_seconds_limit': 3200, 'model_invocations': 0,
                 'jax': '0.4.38', 'flax': '0.10.2',
                 'dependency_note': 'Flax 0.10.2 compatible reconstruction override; source metadata asks >=0.10.4; not original author lock.'})}
    for name, data in files.items():
        path = STATE / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f'Existing immutable artifact differs: {name}')
        else:
            exclusive_write(path, data)
    print(json.dumps({'prepared': True, 'actor_tensors': len(tensors), 'tensor_sha256': tensor_digest(tensors),
                      'all_rounding_matches': all(item['rounding_matches'] for item in evidence.values())}), flush=True)


def build_evaluator():
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(KINETIX))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.models.action_spaces import MultiDiscreteActionDistribution
    from kinetix.util.saving import load_from_json_file
    if jax.__version__ != '0.4.38' or any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('Requires JAX 0.4.38 local CPU')
    initial, static, params = load_from_json_file(str(SCENE))
    static = static.replace(frame_skip=2)
    params = params.replace(max_timesteps=256, dense_reward_scale=0.2)
    initial = initial.replace(timestep=0, last_distance=-1.0)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT,
                           None, params, static, auto_reset=False)
    observation = np.asarray(env.get_obs(initial))
    if observation.shape != (679,) or not np.isfinite(observation).all():
        raise ValueError('Initial observation does not match prompt')
    if not np.allclose(observation[52:54], [2.8, 0.725], atol=0.002):
        raise ValueError('Scene body position mismatch')
    if not np.allclose(observation[234:236], [2.537, 0.735], atol=0.002):
        raise ValueError('Scene green wheel mismatch')
    if (static.num_polygons, static.num_circles, static.num_joints, static.num_thrusters) != (12, 4, 6, 2):
        raise ValueError('Scene shape slots mismatch')

    def logits(weights, obs):
        obs = jnp.where(jnp.isfinite(obs), jnp.clip(obs, -10, 10), 0).astype(jnp.float32)
        hidden = jax.nn.relu(obs @ weights['embed.W'] + weights['embed.b'])
        for layer in range(5):
            hidden = jnp.tanh(hidden @ weights[f'actor{layer}.W'] + weights[f'actor{layer}.b'])
        return hidden @ weights['actor_head.W'] + weights['actor_head.b']

    def episode(weights, episode_id):
        key = jax.random.fold_in(jax.random.PRNGKey(PROTOCOL['seed']), episode_id)
        carry = (initial, key, jnp.array(False), jnp.int32(0), jnp.int32(0), jnp.float32(0))
        def step(carry, step_index):
            state, key, ended, outcome, length, total_return = carry
            key, action_key, env_key = jax.random.split(key, 3)
            obs = env.get_obs(state)
            policy_logits = logits(weights, obs)
            action = MultiDiscreteActionDistribution(policy_logits, [3, 3, 3, 3, 2, 2]).sample(seed=action_key)
            _, advanced, reward, done, info = env.step(env_key, state, action, params)
            finite = jax.tree.reduce(jnp.logical_and, jax.tree.map(lambda value: jnp.isfinite(value).all(), advanced), True)
            # GoalR is sparse contact success; a positive dense reward is not success.
            distance_delta = jnp.where(state.last_distance == -1, 0., -(info['distance'] - state.last_distance) * 0.2)
            sparse = reward - jnp.nan_to_num(distance_delta, nan=0., posinf=0., neginf=0.)
            red = sparse < -0.5
            code = jnp.where(red, 2, jnp.where(info['GoalR'], 1, jnp.where(~finite, 4, 3)))
            newly_done = ~ended & (done | ~finite)
            outcome = jnp.where(newly_done, code, outcome)
            total_return += jnp.where(ended, 0., reward)
            length += (~ended).astype(jnp.int32)
            state = jax.tree.map(lambda old, new: jnp.where(ended, old, new), state, advanced)
            trace = jnp.concatenate((state.polygon.position[5], jnp.array([state.polygon.rotation[5]]),
                                      state.circle.position[0], state.circle.position[1],
                                      action[:1].astype(jnp.float32), jax.nn.softmax(policy_logits[:3])))
            return (state, key, ended | newly_done, outcome, length, total_return), trace
        final, trajectory = jax.lax.scan(step, carry, jnp.arange(256))
        return (final[3], final[4], final[5], trajectory[::16])
    evaluate = jax.jit(jax.vmap(episode, in_axes=(None, 0)))
    metadata = {'jax': jax.__version__, 'devices': [str(device) for device in jax.devices()],
                'static': {key: getattr(static, key) for key in ('frame_skip', 'num_polygons', 'num_circles', 'num_joints', 'num_thrusters')},
                'initial_observation_shape': list(observation.shape), 'initial_body_xy': observation[52:54].tolist(),
                'initial_green_xy': observation[234:236].tolist(), 'initial_motor_probabilities': np.asarray(jax.nn.softmax(logits(load_weights(STATE / 'rounded-original.npz'), jnp.asarray(observation))[:3])).tolist(),
                'packages': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'distrax', 'chex', 'jax2d', 'gymnax')}}
    return evaluate, metadata


def load_weights(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files}
    if set(tensors) != set(SHAPES):
        raise ValueError('Actor archive key mismatch')
    for name, value in tensors.items():
        if value.shape != SHAPES[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Actor archive schema mismatch')
    return tensors


def evaluate_policy(count, batch_size, edited):
    protocol = load_protocol()
    manifest = json.loads((STATE / 'manifest.json').read_text())
    if digest(SCENE.read_bytes()) != manifest['scene_sha256']:
        raise ValueError('Scene hash changed')
    original, _ = reconstruct((SOURCE / 'prompt.txt').read_bytes())
    if tensor_digest(load_weights(STATE / 'rounded-original.npz')) != tensor_digest(original):
        raise ValueError('Rounded original changed')
    path = STATE / ('edited.npz' if edited else 'rounded-original.npz')
    weights = load_weights(path)
    if edited:
        freeze = json.loads((STATE / 'freeze.json').read_text())
        if freeze['status'] != 'frozen' or digest(path.read_bytes()) != freeze['edited_archive_sha256'] or tensor_digest(weights) != freeze['tensor_sha256']:
            raise ValueError('Frozen edited artifact changed')
        if freeze['protocol_sha256'] != digest(json.dumps(protocol, sort_keys=True).encode()):
            raise ValueError('Frozen protocol mismatch')
        from weight_space.car_policy import apply_reply
        raw_reply = (STATE / 'reply.txt').read_bytes()
        if digest(raw_reply) != freeze['reply_sha256']:
            raise ValueError('Frozen first reply changed')
        reapplied, audit = apply_reply(original, raw_reply.decode('utf-8'))
        if not audit['nonidentity'] or tensor_digest(reapplied) != tensor_digest(weights):
            raise ValueError('Edited weights do not match the frozen first reply')
        baseline = json.loads((STATE / f'baseline-{count}.json').read_text())
        if baseline['protocol_sha256'] != freeze['protocol_sha256'] or baseline['episode_ids'] != list(range(count)):
            raise ValueError('No matching preregistered baseline')
    import numpy as np
    def deadline_expired(signum, frame):
        raise TimeoutError('Local CPU evaluation hard deadline exceeded')
    signal.signal(signal.SIGALRM, deadline_expired)
    signal.alarm(3200)
    start = time.monotonic()
    evaluator, metadata = build_evaluator()
    outcomes, lengths, returns, trace = [], [], [], []
    for offset in range(0, count, batch_size):
        ids = np.arange(offset, min(offset + batch_size, count), dtype=np.uint32)
        result = evaluator(weights, ids)
        arrays = [np.asarray(value) for value in result]
        outcomes.extend(arrays[0].tolist())
        lengths.extend(arrays[1].tolist())
        returns.extend(arrays[2].tolist())
        if offset == 0:
            trace = arrays[3][0].tolist()
        print(json.dumps({'episodes_finished': len(outcomes), 'elapsed_seconds': time.monotonic() - start}), flush=True)
        if time.monotonic() - start > 3200:
            raise TimeoutError('Local CPU evaluation runtime budget exceeded')
    record = {'episodes': count, 'seed': PROTOCOL['seed'], 'episode_ids': list(range(count)),
              'success': outcomes.count(1), 'red': outcomes.count(2), 'timeout': outcomes.count(3), 'invalid': outcomes.count(4),
              'success_rate': outcomes.count(1) / count, 'mean_length': float(np.mean(lengths)),
              'mean_return': float(np.mean(returns)), 'wall_seconds': time.monotonic() - start,
              'tensor_sha256': tensor_digest(weights), 'protocol_sha256': digest(json.dumps(protocol, sort_keys=True).encode()),
              'outcomes': outcomes, 'lengths': lengths, 'returns': returns, 'episode_0_trace_every_16_after_step': trace,
              'runtime': metadata, 'paper_bit_exact': False, 'model_called': False}
    target = STATE / f'{"edited" if edited else "baseline"}-{count}.json'
    exclusive_write(target, json_bytes(record))
    if edited:
        baseline_outcomes = np.asarray(baseline['outcomes'])
        edited_outcomes = np.asarray(outcomes)
        paired = {'episodes': count, 'seed': protocol['seed'], 'common_random_numbers': True,
                  'baseline_success': baseline['success'], 'edited_success': record['success'],
                  'success_rate_difference': record['success_rate'] - baseline['success_rate'],
                  'gained_success': int(np.sum((baseline_outcomes != 1) & (edited_outcomes == 1))),
                  'lost_success': int(np.sum((baseline_outcomes == 1) & (edited_outcomes != 1))),
                  'freeze_reply_sha256': freeze['reply_sha256'], 'protocol_sha256': freeze['protocol_sha256']}
        exclusive_write(STATE / f'paired-{count}.json', json_bytes(paired))
    signal.alarm(0)
    print(json.dumps({key: record[key] for key in ('episodes', 'success', 'red', 'timeout', 'invalid', 'mean_return', 'wall_seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'baseline', 'freeze', 'evaluate-edited'])
    parser.add_argument('--episodes', type=int, choices=[32, 1024], default=32)
    parser.add_argument('--batch-size', type=int, choices=[1, 8, 16, 32], default=32)
    parser.add_argument('--reply-file', type=Path)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare()
    elif args.operation == 'freeze':
        if args.reply_file is None:
            parser.error('--reply-file required')
        original, _ = reconstruct((SOURCE / 'prompt.txt').read_bytes())
        print(json.dumps(freeze_reply(STATE, original, args.reply_file.read_bytes(), load_protocol())))
    else:
        evaluate_policy(args.episodes, args.batch_size, args.operation == 'evaluate-edited')


if __name__ == '__main__':
    main()
