import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
from matplotlib.patches import Polygon
from numpy.typing import NDArray


def get_head_yaw(vxs, vys):
    """calculate the yaw of the head"""
    yaws = np.zeros_like(vxs)
    norms = np.sqrt(vxs ** 2 + vys ** 2)
    mask = norms >= 1e-6
    yaws[mask] = np.arctan2(vys[mask], vxs[mask])
    return yaws


def get_tail_amplitude(t_traj: NDArray, q_traj: NDArray, N: int, dist2tip: float):
    # calculate swimming direction
    x_start, y_start = q_traj[0, 0], q_traj[0, N]
    x_end, y_end = q_traj[-1, 0], q_traj[-1, N]
    swim_dir = np.atan2(y_start - y_end, x_start - x_end)
    sin_swim_dir = np.sin(swim_dir)
    cos_swim_dir = np.cos(swim_dir)

    max_tail_amp = 0.
    min_tail_amp = np.inf
    for i, t in enumerate(t_traj):
        # tail position
        x_cur, y_cur, theta_cur = q_traj[i, N - 1], q_traj[i, 2 * N - 1], q_traj[i, 5 * N - 1]
        x_tip = x_cur + dist2tip * np.cos(theta_cur)
        y_tip = y_cur + dist2tip * np.sin(theta_cur)
        dx, dy = x_tip - x_start, y_tip - y_start
        # rotate clockwise
        y_new = -dx * sin_swim_dir + dy * cos_swim_dir
        max_tail_amp = y_new if y_new > max_tail_amp else max_tail_amp
        min_tail_amp = y_new if y_new < min_tail_amp else min_tail_amp
    return max_tail_amp - min_tail_amp


def viz_q_traj(t_traj: NDArray, q_traj: NDArray, N: int, geometry, links, freq: float, video_name=None):
    """animation"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_aspect('equal')

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Arial'

    length = np.asarray(links.length)
    L = sum(length)
    lc = np.asarray(links.lc)

    x_min = np.min(q_traj[:, 0]) - L
    x_max = np.max(q_traj[:, 0]) + L
    y_min = np.min(q_traj[:, N]) - L / 2
    y_max = np.max(q_traj[:, N]) + L / 2
    y_lim = max(abs(y_min), abs(y_max))

    # video writer setup
    if video_name:
        if os.path.exists(video_name):
            os.remove(video_name)
        
        frame_rate = len(t_traj) / (t_traj[-1] - t_traj[0])
        writer = animation.FFMpegWriter(fps=int(frame_rate))
    
    def update(frame):
        ax.clear()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([-y_lim, y_lim])
        
        thetas = q_traj[frame, 4 * N: 5 * N]
        # draw midline
        nodes = np.zeros((2, N + 1))
        nodes[0, 0] = q_traj[frame, 0] - lc[0] * np.cos(thetas[0])
        nodes[1, 0] = q_traj[frame, N] - lc[0] * np.sin(thetas[0])
        for j in range(N):
            nodes[0, j + 1] = nodes[0, j] + length[j] * np.cos(thetas[j])
            nodes[1, j + 1] = nodes[1, j] + length[j] * np.sin(thetas[j])
        
        # ax.plot(nodes[0, :], nodes[1, :], 'k-')

        # plot shape
        for j in range(N):
            l_start = np.sum(length[:j])
            l_end = np.sum(length[:j + 1])
            num_points = int(round((l_end - l_start) / 0.005)) + 1
            ls = np.linspace(l_start, l_end, num_points)
            rs = geometry.calc_r(ls)
            
            shape_node = np.zeros((2, len(ls) * 2))
            shape_node[0, :len(ls)] = ls - l_start
            shape_node[0, len(ls):] = np.flip(ls - l_start)
            shape_node[1, :len(ls)] = rs
            shape_node[1, len(ls):] = -np.flip(rs)
            # transform
            delta_q = nodes[:, j]
            RM = np.asarray([[np.cos(thetas[j]), -np.sin(thetas[j])], 
                           [np.sin(thetas[j]), np.cos(thetas[j])]])
            for k in range(shape_node.shape[1]):
                shape_node[:, k] = delta_q + RM @ shape_node[:, k]
            
            shape_j = Polygon(shape_node.T, closed=True, edgecolor='k', facecolor='#1f77b4', alpha=0.5)
            ax.add_patch(shape_j)

        # plot trajectory
        # ax.plot(q_traj[:frame + 1, 0], q_traj[:frame + 1, N], 'k-', linewidth=2.)
        x_traj = q_traj[:frame + 1, 0]
        y_traj = q_traj[:frame + 1, N]
        points = np.asarray([x_traj, y_traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a colormap with constant color but varying alpha
        n_segments = len(segments)
        alphas = np.linspace(0.1, 1.0, n_segments)  # from transparent to opaque
        colors = np.zeros((n_segments, 4))  # RGBA
        colors[:, :3] = 0.0  # black RGB
        colors[:, 3] = alphas  # varying alpha
        # Create LineCollection
        traj_lines = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(traj_lines)

        # print time
        plt.text(0.95, 0.95, f"t = {t_traj[frame] - t_traj[0]:.2f} s", 
            transform=plt.gca().transAxes, 
            ha='right', va='top',
            fontsize=24, fontweight='bold')

        # axis
        ax.set_xlabel('x (BL)', fontsize=32)
        ax.set_ylabel('y (BL)', fontsize=32)
        ax.tick_params(axis='both', labelsize=32)
        x_lim = int(-x_min / L) + 1
        ax.set_xticks([-i * L for i in range(-1, x_lim,2)])
        ax.set_xticklabels([i for i in range(1, -x_lim, -2)])
        ax.set_yticks([-.5 * L, 0, .5 * L])
        ax.set_yticklabels([-.5, 0, .5])

    ani = animation.FuncAnimation(fig, update, frames=len(t_traj), repeat=False)

    if video_name:
        ani.save(video_name, writer=writer, dpi=300)
    else:
        plt.show()   

    # # save last frame
    # update(len(t_traj) - 1)
    # fig.tight_layout()
    # fig.savefig('last_frame.pdf', format='pdf')
    # plt.close(fig) 


def plot_figs(N: int, t_traj: NDArray, q_traj: NDArray, L: float, freq: float):
    """plot trajectory, velocity, and head yaw"""
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    _, axs = plt.subplots(3, 1, figsize=(4, 10))

    # =========================
    # Plot trajectory
    # =========================
    axs[0].plot(q_traj[:, 0], q_traj[:, N], linewidth=1.5)
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')
    axs[0].set_title('Trajectory')

    # =========================
    # Velocity calculation
    # =========================
    cmap = plt.get_cmap('magma')
    vx_color = cmap(0.25)
    vy_color = cmap(0.5)
    v_color  = cmap(0.75)

    vxs_g = q_traj[:, 2 * N]
    vys_g = q_traj[:, 3 * N]
    thetas = q_traj[:, 4 * N]

    # Transform to head frame
    vxs = -vxs_g * np.cos(thetas) - vys_g * np.sin(thetas)
    vys =  vxs_g * np.sin(thetas) - vys_g * np.cos(thetas)
    vel = np.sqrt(vxs_g ** 2 + vys_g ** 2)

    # Nondimensional quantities
    t_star = t_traj * freq
    vx_BL_s = vxs / L
    vy_BL_s = vys / L
    v_BL_s = vel / L

    # =========================
    # Save velocity data
    # =========================
    os.makedirs("res", exist_ok=True)

    output_file = f"res/velocity_data_freq_{freq:.1f}.csv"

    output_data = np.column_stack([
        t_star,
        vx_BL_s,
        vy_BL_s,
        v_BL_s
    ])

    np.savetxt(
        output_file,
        output_data,
        delimiter=",",
        header="t_star,vx_BL_s,vy_BL_s,v_BL_s",
        comments=""
    )

    print(f"Saved velocity data to: {output_file}")

    # =========================
    # Plot velocity
    # =========================
    axs[1].plot(t_star, vx_BL_s, linewidth=2., color=vx_color, label=r'$v_x$')
    axs[1].plot(t_star, vy_BL_s, linewidth=2., color=vy_color, label=r'$v_y$')
    axs[1].plot(t_star, v_BL_s, linewidth=2., color=v_color, label=r'$v$')

    axs[1].set_xlabel(r'$t^*$', fontsize=24)
    axs[1].set_ylabel(r'$V$ (BL/s)', fontsize=24)
    axs[1].tick_params(axis='both', labelsize=24)
    axs[1].set_xticks([0, 20])
    axs[1].set_yticks([0, 0.25])
    axs[1].legend(loc='best', fontsize=18)

    # =========================
    # Plot head yaw
    # =========================
    yaws = get_head_yaw(vxs, vys)
    axs[2].plot(t_traj, yaws, linewidth=1.5)
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Yaw (rad)')
    axs[2].set_title('Head Yaw')

    plt.tight_layout()
    plt.show()


def calc_efficiency(N, m_total, t_traj, q_traj, M_rec):
    """Compute COT=P/mgU"""
    t_eff = t_traj[len(t_traj) // 2:]

    dphi1_eff = q_traj[len(q_traj) // 2:, 5 * N + 1] - q_traj[len(q_traj) // 2:, 5 * N]

    # input power
    P_in = np.trapezoid(M_rec[len(M_rec) // 2:] * dphi1_eff, t_eff)
    P_in /= (t_eff[-1] - t_eff[0])

    # thrust power
    vxs_g = q_traj[:, 2 * N]
    vys_g = q_traj[:, 3 * N]
    vel = np.sqrt(vxs_g ** 2 + vys_g ** 2)
    v_stable = float(np.mean(vel[len(vel) // 2: ]))
    # Ek = 0.5 * m_total * v_stable * v_stable

    if (not np.isfinite(P_in)) or P_in < 0 or (not np.isfinite(v_stable)) or v_stable < 1e-8:
        return np.inf

    return P_in / (m_total * 9.8 * v_stable)


def plot_midline_wave(t_traj: NDArray, q_traj: NDArray, freq: float, N: int, links):
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    _, ax = plt.subplots(figsize=(12, 6))

    t_eff = t_traj[len(t_traj)-400:]
    q_traj_eff = q_traj[len(t_traj)-400:, :]
    plot_interval = len(t_eff) // int((t_eff[-1] - t_eff[0]) * 4 * freq)

    # calculate swimming direction
    x_start, y_start = q_traj[0, 0], q_traj[0, N]
    x_end, y_end = q_traj[-1, 0], q_traj[-1, N]
    swim_dir = np.atan2(y_start - y_end, x_start - x_end)
    sin_swim_dir = np.sin(swim_dir)
    cos_swim_dir = np.cos(swim_dir)

    Ys = []
    Xs = []
    for i, _ in enumerate(t_eff):
        if i % plot_interval == 0:
            # extract midline
            nodes = np.zeros((2, N + 1))
            nodes[0, 0] = q_traj_eff[i, 0] - links.lc[0] * np.cos(q_traj_eff[i, 4 * N])
            nodes[1, 0] = q_traj_eff[i, N] - links.lc[0] * np.sin(q_traj_eff[i, 4 * N])
            for j in range(N):
                nodes[0, j + 1] = nodes[0, j] + links.length[j] * np.cos(q_traj_eff[i, 4 * N + j])
                nodes[1, j + 1] = nodes[1, j] + links.length[j] * np.sin(q_traj_eff[i, 4 * N + j])
            
            # transform
            dist2zero = 0.
            for j in range(N + 1):
                dx, dy = nodes[0, j] - x_start, nodes[1, j] - y_start
                # rotate clockwise
                x_new = dx * cos_swim_dir + dy * sin_swim_dir
                y_new = -dx * sin_swim_dir + dy * cos_swim_dir
                if j == 0:
                    # dist2zero = -x_new
                    dist2zero = links.lc[0] + np.sqrt((q_traj_eff[i, 0] - links.lc[0]) ** 2 + q_traj_eff[i, N] ** 2)
                nodes[0, j], nodes[1, j] = x_new + dist2zero, y_new
            
            Xs.append(nodes[0, :].copy())
            Ys.append(nodes[1, :].copy())
            plt.plot(nodes[0, :], nodes[1, :], 'k-', linewidth=1.)
    
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    upper = Y.max(axis=0)
    lower = Y.min(axis=0)
    idx_upper = np.argmax(Y, axis=0)
    idx_lower = np.argmin(Y, axis=0)
    xs_upper = X[idx_upper, np.arange(Y.shape[1])]
    xs_lower = X[idx_lower, np.arange(Y.shape[1])]
    ax.plot(xs_upper, upper, 'k-', linewidth=4.)
    ax.plot(xs_lower, lower, 'k-', linewidth=4.)
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)", fontsize=28)
    ax.set_ylabel("y (m)", fontsize=28)
    ax.tick_params(axis='both', labelsize=24)

    body_len = np.sum(links.length)
    ax.set_xlim(0, body_len)
    ax.set_ylim(-0.2 * body_len, 0.2 * body_len)
    plt.show()
