import numpy as np
import pybullet as p

from gymnasium import spaces
from myAviary import MyAviary

class SingleAviary(MyAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ghi đè chỉ mục tiêu cho con Agent (agent_idx)
        self.agent_idx = 0 # Mặc định điều khiển con đầu tiên nếu không có bridge

    def _actionSpace(self):
        # SB3 cần (3,) thay vì (N, 3) cho Single Agent
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _observationSpace(self):
        # Trả về state của con Agent duy nhất
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _computeObs(self):
        # Chỉ lấy Obs của con drone mà AI đang điều khiển
        full_obs = super()._computeObs() 
        return full_obs[self.agent_idx].astype(np.float32)

    def step(self, action):
        # Chuyển action (3,) của SB3 thành action (N, 3) cho MyAviary
        action_all = np.zeros((self.num_drones, 3))
        action_all[self.agent_idx] = action # Chỉ con Agent di chuyển theo AI
        
        # Các con khác đứng yên (action = 0)
        rpm_all = self._apply_physics_step(action_all)
        
        # Chạy bước vật lý
        _, _, _, _, info = super(MyAviary, self).step(rpm_all)
        
        obs = self._computeObs()
        reward = self._computeReward()
        
        # Thêm Reward dẫn đường (Shaping) cho Single Agent để học nhanh hơn
        dist_to_hole = np.linalg.norm(obs[0:3] - self.hole_position)
        reward -= dist_to_hole * 0.1 # Càng xa lỗ càng bị trừ điểm
        
        done = np.all(self.battery <= 0)
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        if self.baseRandom: self.base_A, self.base_B = self._random_bases()
        if self.windRandom: self._random_wind()
        if self.droneRandom: self._init_drones_bridge()
        else: self.agent_idx = 0 # Fallback
            
        self.battery = np.ones(self.num_drones)
        return super().reset(seed=seed, options=options)