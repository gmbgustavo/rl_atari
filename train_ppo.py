from statistics import mean, median

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO


class Pixels84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation):
        image = Image.fromarray(observation).convert("L")
        image = image.resize((84, 84), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)[..., None]


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
    env = gym.wrappers.TimeLimit(Pixels84(env), max_episode_steps=1800)
    rng = np.random.default_rng(42)
    avaliar(env, lambda obs: int(rng.integers(env.action_space.n)), "Aleatório")

    model = PPO(
        "CnnPolicy", env, device="cuda",
        n_steps=1024, batch_size=64, seed=42, verbose=1,
    )
    model.learn(total_timesteps=32_768)
    model.save("ppo_space_invaders")

    avaliar(
        env,
        lambda obs: int(model.predict(obs, deterministic=False)[0].item()),
        "PPO",
    )