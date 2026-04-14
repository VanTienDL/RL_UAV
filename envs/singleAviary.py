import numpy as np
import pybullet as p

from gymnasium import spaces
from .myAviary import MyAviary

class SingleAviary(MyAviary):
    def __init__(self, agent_idx=0,**kwargs):
        #kwargs nghĩa là mọi tham số có đặt tên, nên gọi env này có thể gui=True, num_drone=3... nói chung là gọi có thể có hoặc default mọi tham số của lớp cha MyAviary

        super().__init__(**kwargs) #Gọi lớp MyAviary tạo drone, cột, gió,... bằng cách giải nén kwargs truyền vào SingleAviary

        # Ghi đè chỉ mục tiêu cho con Agent (agent_idx)
        self.agent_idx = agent_idx # Mặc định điều khiển con đầu tiên nếu không có bridge

    def _actionSpace(self):
        # SB3 cần (3,) thay vì (N, 3) cho Single Agent
        # Action chỉ bao gồm 1 vector tọa độ tới (x,y,z) thực
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _observationSpace(self):
        # Trả về state của con Agent duy nhất dưới dạng 1 vector 7 chiều
        # Gồm 3 tọa độ hiện tại, 3 vận tốc, 1 pin 
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _computeObs(self): # Lớp cha MyAviary trả ra state mới ma trận n drone
        # Chỉ lấy state mới của drone hàng thứ agent_idx
        full_obs = super()._computeObs() 
        return full_obs[self.agent_idx].astype(np.float32)

    def step(self, action):
        # Mọi drone khác đều action (0,0,0) vì pos + (0,0,0)*0,2 = pos
        action_all = np.zeros((self.num_drones, 3)) #Toàn drone action (0,0,0)
        action_all[self.agent_idx] = action # agent_idx mới có (x,y,z)
        
        # 1. Tính RPM từ lớp cha
        rpm_all = self._apply_physics_step(action_all)
        
        # 2. Đẩy vào môi trường gốc (ông nội CtrlAviary)
        # super(MyAviary, self) là gọi CtrlAviary để hành động lên Pybullet
        obs, reward, terminated, truncated, info = super(MyAviary, self).step(rpm_all)
        
        # 3. Ghi đè bằng logic riêng của Single Agent
        custom_obs = self._computeObs()
        custom_reward = self._computeReward()
        
        # Thêm reward dẫn đường
        dist = np.linalg.norm(custom_obs[0:3] - self.hole_position)
        custom_reward -= dist * 0.1
        
        return custom_obs, custom_reward, self._computeTerminated(), False, info

    def reset(self, seed=None, options=None):
        if self.baseRandom: self.base_A, self.base_B = self._random_bases()
        if self.windRandom: self._random_wind()
        if self.droneRandom: self._init_drones_bridge()
        else: self.agent_idx = 0 # Fallback
            
        self.battery = np.ones(self.num_drones)
        return super().reset(seed=seed, options=options)