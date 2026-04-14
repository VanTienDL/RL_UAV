import os
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import class của cậu
from envs.singleAviary import SingleAviary 

def train():
    # 1. Tạo thư mục lưu trữ kết quả
    filename = datetime.now().strftime("single_drone_%Y%m%d_%H%M%S")
    log_dir = "./logs/" + filename + "/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Khởi tạo môi trường
    # Cậu có thể tùy chỉnh num_drones và gui ở đây
    env = SingleAviary(
        agent_idx=0,
        num_drones=3,         # Tổng số drone (1 con học, 2 con đứng im)
        comm_radius=2.0,      # Bán kính kết nối
        gui=False,            # Tắt GUI khi train để nhanh hơn (bật True nếu muốn xem)
        baseRandom=True,      # Random trạm A, B để AI học cách thích nghi
        droneRandom=True,     # Đội hình bridge random để tạo "lỗ hổng" khác nhau
        windRandom=True       # Thêm gió cho "khó"
    )

    # Wrap môi trường để SB3 có thể ghi log
    env = Monitor(env, log_dir)

    # 3. Định nghĩa mạng Neuron (Mô hình PPO)
    # Chúng ta dùng mạng MLP (Multi-Layer Perceptron) với 2 lớp ẩn, mỗi lớp 256 node
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,       # Tốc độ học
        n_steps=2048,             # Số bước thu thập dữ liệu trước khi cập nhật mạng
        batch_size=64,            # Kích thước batch khi train
        n_epochs=10,              # Số lần học lại trên cùng một tập dữ liệu
        gamma=0.99,               # Hệ số chiết khấu (tầm nhìn xa của AI)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,            # Khuyến khích khám phá (Exploration)
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,  # Để xem biểu đồ trên Tensorboard
        verbose=1
    )

    # 4. Callbacks: Tự động lưu mô hình mỗi 10,000 bước
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="ppo_drone_model"
    )

    # 5. Bắt đầu huấn luyện
    print("--- Bắt đầu huấn luyện ---")
    try:
        model.learn(
            total_timesteps=200000,   # Cậu có thể tăng lên 1,000,000 để AI khôn hơn
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("--- Đã dừng huấn luyện thủ công ---")

    # 6. Lưu mô hình cuối cùng
    model.save(log_dir + "ppo_single_drone_final")
    print(f"Mô hình đã được lưu tại: {log_dir}")

if __name__ == "__main__":
    train()