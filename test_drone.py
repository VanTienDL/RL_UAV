import time
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

env = CtrlAviary(
    drone_model=DroneModel.CF2X, 
    num_drones=1,                
    physics=Physics.PYB,         
    gui=True                     
)

obs, info = env.reset()

# Tăng lên 5000 để chạy được khoảng 20 giây
for i in range(5000):
    # Dùng lực đẩy 0.50001 (vừa đủ thắng trọng lực một chút để bay từ từ)
    # Hoặc dùng 0.6 để nó bay vèo lên cho dễ thấy
    action = np.array([[0, 0, 0, 0.6]]) 
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    
    # Nếu máy cậu mạnh, nó chạy vèo cái là hết 5000 step. 
    # Hãy để sleep lâu hơn một chút để mắt người kịp nhìn.
    time.sleep(1/60) 

env.close()