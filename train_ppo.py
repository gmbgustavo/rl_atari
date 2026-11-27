from statistics import mean, median

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from collections import deque
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT = Path("checkpoints/ppo_sinvaders_4frames_latest.zip")


class SaveCheckpointCallback(BaseCallback):
    def __init__(self, path, save_freq=10_000):
        super().__init__()
        self.path = path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(str(self.path))
            print(f"Checkpoint salvo em {self.num_timesteps} etapas.")
        return True


class Pixels84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation):
        image = Image.fromarray(observation).convert("L")
        image = image.resize((84, 84), Image.Resampling.BILINEAR)
        return np.array(image, dtype=np.uint8, copy=True)[..., None]


def avaliar(env, escolher_acao, nome, partidas=20):
    resultados = []

    for _ in range(partidas):
        obs, _ = env.reset()
        total = 0.0

        while True:
            obs, reward, terminated, truncated, _ = env.step(escolher_acao(obs))
            total += reward
            if terminated or truncated:
                break

        resultados.append(total)

    print(
        f"{nome}: média={mean(resultados):.1f} | "
        f"mediana={median(resultados):.1f} | melhor={max(resultados):.0f}"
    )


def treinar(env):
    env = gym.wrappers.TimeLimit(
        FrameStack(Pixels84(env), stack_size=4),
        max_episode_steps=1800,
    )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    retomando = CHECKPOINT.exists()

    if retomando:
        print(f"Continuando de {CHECKPOINT}")
        model = PPO.load(CHECKPOINT, env=env, device="cuda")
    else:
        model = PPO(
            "CnnPolicy", env, device="cuda",
            n_steps=1024, batch_size=64, seed=42, verbose=1,
        )

    callback = SaveCheckpointCallback(CHECKPOINT)

    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=callback,
            reset_num_timesteps=not retomando,
        )
    except KeyboardInterrupt:
        model.save(str(CHECKPOINT))
        print(f"\nTreino interrompido. Checkpoint salvo em {CHECKPOINT}")
        return

    model.save(str(CHECKPOINT))
    print(f"Treino concluído. Checkpoint salvo em {CHECKPOINT}")

    avaliar(
        env,
        lambda obs: int(model.predict(obs, deterministic=True)[0].item()),
        "PPO",
    )

class FrameStack(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

        height, width, channels = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, channels * stack_size),
            dtype=np.uint8,
        )

    def _stack(self):
        return np.concatenate(tuple(self.frames), axis=-1).copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        # Preenche o histórico inicial repetindo a primeira imagem.
        for _ in range(self.stack_size):
            self.frames.append(obs.copy())
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs.copy())
        return self._stack(), reward, terminated, truncated, info