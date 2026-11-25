import gymnasium as gym
import numpy as np


class SpaceInvadersEnv(gym.Env):
    def __init__(self, reset_game, advance, read_stats, frame_to_rgb):
        super().__init__()
        self.reset_game = reset_game
        self.advance = advance
        self.read_stats = read_stats
        self.frame_to_rgb = frame_to_rgb

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(210, 160, 3), dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        frame = self.reset_game()
        self.score, lives, _ = self.read_stats()
        return self.frame_to_rgb(frame).copy(), {"score": self.score, "lives": lives}

    def step(self, action):
        frame = self.advance(int(action))
        score, lives, game_over = self.read_stats()
        reward = (score - self.score) % 10_000  # cobre a virada de 9999 para 0000
        self.score = score

        obs = self.frame_to_rgb(frame).copy()
        info = {"score": score, "lives": lives}
        return obs, float(reward), bool(game_over), False, info