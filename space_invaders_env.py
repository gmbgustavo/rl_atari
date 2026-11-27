from collections import deque
from contextlib import ExitStack

import gymnasium as gym
import numpy as np

from stella import ACTIONS, FIRE, Stella

FRAMES_POR_ACAO = 4
LIMITE_PASSOS = 1800
PRIMEIRA_SEMENTE_AVALIACAO = 10_000
ACAO_TIRO = ACTIONS.index(FIRE)


class SpaceInvadersEnv(gym.Env):
    def __init__(self, core, rom, *, legado=False, render=False):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8)
        self.frames = deque(maxlen=4)
        self.legado = legado

        with ExitStack() as recursos:
            self.stella = recursos.enter_context(Stella(core, render=render))
            self.stella.load_rom(rom)
            self.recursos = recursos.pop_all()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        noops = 0 if self.legado else int(self.np_random.integers(0, 31))
        self.stella.reset(hard=not self.legado, noops=noops)
        self.score, lives, game_over = self.stella.stats()
        if (self.score, lives, game_over) != (0, 3, False):
            raise RuntimeError(f"Reset inesperado: {(self.score, lives, game_over)}")

        self.frames.clear()
        self.frames.extend([self.stella.observation()] * self.frames.maxlen)
        return self.observation(), {"score": self.score, "lives": lives}

    def observation(self):
        return np.concatenate(tuple(self.frames), axis=-1)

    def step(self, action):
        buttons = ACTIONS[int(action)]
        if self.legado:
            self.stella.run_frames(40, buttons)
            frame = self.stella.observation()
        else:
            self.stella.run_frames(FRAMES_POR_ACAO - 1, buttons)
            penultimo = self.stella.observation()
            self.stella.run_frames(1, buttons)
            # Objetos piscam no Atari; combine os dois quadros mais recentes.
            frame = np.maximum(penultimo, self.stella.observation())

        score, lives, game_over = self.stella.stats()
        reward = (score - self.score) % 10_000  # cobre a virada do placar
        self.score = score
        self.frames.append(frame)
        return self.observation(), float(reward), game_over, False, {
            "score": score, "lives": lives,
        }

    def close(self):
        self.recursos.close()


def criar_ambiente(core, rom, **kwargs):
    return gym.wrappers.TimeLimit(SpaceInvadersEnv(core, rom, **kwargs), LIMITE_PASSOS)


def jogar_partida(env, escolher_acao, *, seed):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    passos = 0
    while True:
        obs, reward, terminou, truncou, info = env.step(escolher_acao(obs))
        total += reward
        passos += 1
        if terminou or truncou:
            return total, passos, terminou, info
