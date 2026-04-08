import numpy as np
import pybullet as p

from gymnasium import spaces
from myAviary import MyAviary

class MultiAviary(MyAviary):
    def _actionSpace(self):
        # MAPPO thường nhận (num_drones, 3)
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_drones, 3), dtype=np.float32)

    def _observationSpace(self):
        obs_dim = 3 + 3 + 1
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_drones, obs_dim), dtype=np.float32)

    def step(self, action):
        # Action lúc này đã là ma trận (N, 3) từ thuật toán MAPPO
        rpm_all = self._apply_physics_step(action)
        super(MyAviary, self).step(rpm_all)

        obs = self._computeObs()
        reward = self._computeReward() # Reward chung cho cả đàn (Global Reward)
        
        done = np.all(self.battery <= 0)
        return obs, reward, done, False, {}