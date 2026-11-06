"""Render one paired episode of the original and guided-Astra car policies as a GIF."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import car_oneshot as base

GUIDED = ROOT / 'artifacts/car-astra-guided/edited.npz'
OUTPUT = ROOT / 'docs/car-fix.gif'
EPISODE = 0


def rollout(weights):
    import jax
    import jax.numpy as jnp
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.models.action_spaces import MultiDiscreteActionDistribution
    from kinetix.render.renderer_pixels import make_render_pixels
    from kinetix.util.saving import load_from_json_file
    initial, static, params = load_from_json_file(str(base.SCENE))
    static = static.replace(frame_skip=2)
    params = params.replace(max_timesteps=256, dense_reward_scale=0.2)
    state = initial.replace(timestep=0, last_distance=-1.0)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT,
                           None, params, static, auto_reset=False)
    render = jax.jit(make_render_pixels(params, static.replace(downscale=1)))  # display only
    weights = {name: jnp.asarray(value) for name, value in weights.items()}

    def logits(obs):
        obs = jnp.where(jnp.isfinite(obs), jnp.clip(obs, -10, 10), 0).astype(jnp.float32)
        hidden = jax.nn.relu(obs @ weights['embed.W'] + weights['embed.b'])
        for layer in range(5):
            hidden = jnp.tanh(hidden @ weights[f'actor{layer}.W'] + weights[f'actor{layer}.b'])
        return hidden @ weights['actor_head.W'] + weights['actor_head.b']

    key = jax.random.fold_in(jax.random.PRNGKey(base.PROTOCOL['seed']), EPISODE)
    frames = [render(state)]
    for _ in range(256):
        key, action_key, env_key = jax.random.split(key, 3)
        action = MultiDiscreteActionDistribution(logits(env.get_obs(state)), [3, 3, 3, 3, 2, 2]).sample(seed=action_key)
        _, state, _, done, _ = env.step(env_key, state, action, params)
        frames.append(render(state))
        if bool(done):
            break
    return frames


def to_image(frame, label, colour):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    pixels = np.asarray(frame).transpose(1, 0, 2)[::-1].clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels[300:]).convert('RGB')
    ImageDraw.Draw(image).text((12, 10), label, fill=colour, font=ImageFont.load_default(size=22))
    return image


def main():
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'
    sys.path.insert(0, str(base.KINETIX))
    from PIL import Image
    original = rollout(base.load_weights(base.STATE / 'rounded-original.npz'))
    guided = rollout(base.load_weights(GUIDED))
    count = max(len(original), len(guided)) + 15
    frames = []
    for index in range(count):
        left = to_image(original[min(index, len(original) - 1)], 'Original policy', (200, 0, 0))
        right = to_image(guided[min(index, len(guided) - 1)], 'Astra one-shot edit', (0, 120, 0))
        canvas = Image.new('RGB', (left.width + right.width + 4, left.height), 'white')
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 4, 0))
        frames.append(canvas)
    OUTPUT.parent.mkdir(exist_ok=True)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
    print({'original_steps': len(original) - 1, 'guided_steps': len(guided) - 1, 'output': str(OUTPUT)})


if __name__ == '__main__':
    main()
