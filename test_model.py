import argparse
from pathlib import Path

import numpy as np
from fish_env import FishEnv
from stable_baselines3 import PPO
from fish_viz import viz_q_traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=float, required=True, help="Tail-beat frequency, e.g. 2.0")
    parser.add_argument("--model_dir", type=str, default="res", help="Folder containing saved PPO models")
    parser.add_argument("--video_dir", type=str, default="res/video", help="Folder to save visualization video")
    args = parser.parse_args()

    freq = args.freq
    N = 8
    L = .4

    mu = np.asarray([0.0] +[1e-4] * (N - 2))
    Cf = np.asarray([.1925] * (N - 1))
    Cd = np.asarray([0.2] * (N - 1))
    Cl_fin = 5.1938
    Cd_fin = np.asarray([5, 0.5])

    t_end = 20. if freq < 1. else 10.
    env = FishEnv(L=L, N=N, mu=mu, freq=freq, 
                  Cf=Cf, Cd=Cd, Cl_fin=Cl_fin, Cd_fin=Cd_fin,
                  head_len=0.5, amp=np.deg2rad(25),
                  t_end=t_end,
                  reward_weight=[0., 2., -1.],
                  sim_freq=int(50 * freq))
    
    model_path = Path(args.model_dir) / f"ppo_fish_{freq:.1f}.zip"
    model = PPO.load(model_path, device='cpu')
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    k_value = env.action2sim_inp(action)
    obs, reward, _, _, info = env.step(action)
    if info["q_traj"] is None:
        raise RuntimeError(f"Model evaluation failed: {info['error_message']}")
    vel_stable = info["vel_stable"]
    cot = info["cot"]
    print(f"Action: {action}\n \
        Stiffness: {k_value}\n \
        Velocity: {vel_stable}\n \
        COT: {cot}")


    ### Plot figures
    t_traj = info["t_traj"]
    q_traj = info["q_traj"]

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    viz_q_traj(t_traj, q_traj, N, env.geometry, env.links, freq,
        video_name=str(video_dir / f"fish_{freq:.1f}.mp4"))


if __name__ == "__main__":
    main()
