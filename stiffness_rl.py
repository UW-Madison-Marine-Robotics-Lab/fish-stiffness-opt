import numpy as np
import torch
from fish_env import FishEnv, PPOLoggingCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    N = 8
    L = .4
    freq = 2.

    mu = np.asarray([0.0] +[1e-4] * (N - 2))
    Cf = np.asarray([.1925] * (N - 1))
    Cd = np.asarray([0.2] * (N - 1))
    Cl_fin = 5.1938
    Cd_fin = np.asarray([5, 0.5])

    # parallel environments
    t_end = 20. if freq < 1. else 10.
    def make_env():
        return FishEnv(L=L, N=N, mu=mu, freq=freq, 
                       Cf=Cf, Cd=Cd, Cl_fin=Cl_fin, Cd_fin=Cd_fin,
                       head_len=0.5, amp=np.deg2rad(25),
                       t_end=t_end,
                       reward_weight=[0., 2., -1.],
                       sim_freq=int(50 * freq),
                       )

    num_envs = 64
    vec_env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
    # check_env(env)
 
    # RL Agent
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = PPO("MlpPolicy", 
                vec_env, 
                verbose=1, 
                learning_rate=0.0003, 
                n_steps=32, 
                batch_size=64, 
                n_epochs=10,
                ent_coef=0.005,
                clip_range=0.2,
                tensorboard_log=f"./res/ppo_log_{freq:.1f}/",
                device=torch.device("cpu"))

    ppo_callback = PPOLoggingCallback()
    model.learn(total_timesteps=400000, callback=ppo_callback)

    file_name = f"ppo_fish_{freq:.1f}.zip"
    model.save("res/" + file_name)    
