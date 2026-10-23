"""Run a prescribed-stiffness fish simulation and visualize the trajectory."""

import time
from pathlib import Path

import numpy as np
import fish_sim
from fish_viz import calc_efficiency, viz_q_traj


# Fluid properties
density = 1000.
fluidV = np.asarray([0.0, 0.0])
fluid = fish_sim.Fluid(density, fluidV)

# Fish geometry
L = 0.4
geometry = fish_sim.Geometry(L)

# Links
N = 8
head_len = L * .5
fin_len = L * .2
mid_len = (L - head_len - fin_len) / (N - 2)
link_length = np.asarray([head_len] + [mid_len] * (N - 2) + [fin_len])
links = fish_sim.Link(N, L, link_length, fluid, geometry)

# Hydrodynamic coefficients
Cf = np.asarray([.1925] * (N - 1))
Cd = np.asarray([0.2] * (N - 1))
Cl_fin = 5.1938
Cd_fin = np.asarray([5, 0.5])
hydrodyn = fish_sim.HydroDyn(N, Cf, Cd, Cl_fin, Cd_fin)

# Joint stiffness and actuation
freq = 2.0
k0 = 40
k_values = k0 * np.asarray([
    np.pow(1 - (head_len + mid_len * (i + 1)) / L, 4)
    for i in range(N - 2)
])
k = np.concatenate(([0], k_values))
mu = np.asarray([0.0] +[1e-4] * (N - 2))
joints = fish_sim.Joint(N, k, mu, np.deg2rad(25), freq, 0.)

# Initial state
q_init = np.zeros(4 + 2 * N)
bias = np.deg2rad(0)
q_init[5: N + 4] = np.full(N - 1, joints.amp * np.sin(joints.phase) + bias)
I1 = float(links.Izz[0])
I2 = float(links.Izz[1])
dTheta = joints.amp * (2 * np.pi * joints.freq) * np.cos(joints.phase)
q_init[N + 4] = - I2 * dTheta / (I1 + I2)
q_init[N + 5] = I1 * dTheta / (I1 + I2)
q0 = fish_sim.get_q0(q_init, links)

# Simulation settings
sim_freq = 50
t_start = 0.
t_end = 10.0

# Run simulation
sim = fish_sim.FishSimulator(sim_freq, t_start, t_end, q0, links, joints, hydrodyn)
t1 = time.time()
results = sim.run(120)
t2 = time.time()
print(f'Solving time: {t2 - t1:.3}s')

t_traj = np.asarray(results.t_traj)
q_traj = np.asarray(results.q_traj)
M_rec = np.asarray(results.M_rec)

vxs_g = q_traj[:, 2 * N]
vys_g = q_traj[:, 3 * N]
vel = np.sqrt(vxs_g ** 2 + vys_g ** 2)
print(f'Stable Vel: {float(np.mean(vel[len(vel) // 2: ])) / L}')

eff = calc_efficiency(N, links.m_total, t_traj, q_traj, M_rec)
print(f'Mass: {links.m_total:.4f}')
print(f'COT: {eff:.5f}')

Path('res').mkdir(exist_ok=True)
viz_q_traj(t_traj, q_traj, N, geometry, links, freq, video_name=f'res/example-swimming-{freq:.1f}.mp4')
