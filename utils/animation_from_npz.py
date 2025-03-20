import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

def animation_from_npz(
        path,
        save_path,
        boundaries,
        timestep_stride=5,
        colorful=True,
        follow_taichi_coord=False):

    data = dict(np.load(path, allow_pickle=True))
    for i, (sim, info) in enumerate(data.items()):
        positions = info[0]
    ndim = positions.shape[-1]

    # compute vel magnitude for color bar
    if colorful:
        initial_vel = np.zeros(positions[0].shape)
        initial_vel = initial_vel.reshape((1, initial_vel.shape[0], initial_vel.shape[1]))
        vel = positions[1:] - positions[:-1]
        vel = np.concatenate((initial_vel, vel))
        vel_magnitude = np.linalg.norm(vel, axis=-1)

    if ndim == 2:
        # make animation
        fig, ax = plt.subplots()

        def animate(i):
            fig.clear()
            # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
            ax.set_xlim(boundaries[0][0], boundaries[0][1])
            ax.set_ylim(boundaries[1][0], boundaries[1][1])
            ax.scatter(positions[i][:, 0], positions[i][:, 1], s=1)
            ax.grid(True, which='both')

    if ndim == 3:
        # make animation
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def animate(i):
            print(f"Render step {i}/{len(positions)}")
            fig.clear()

            if colorful:
                cmap = plt.cm.viridis
                vmax = np.ndarray.flatten(vel_magnitude).max()
                vmin = np.ndarray.flatten(vel_magnitude).min()
                sampled_value = vel_magnitude[i]

            if follow_taichi_coord:
                # Note: z and y is interchanged to match taichi coordinate convention.
                ax = fig.add_subplot(projection='3d', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[2][0], boundaries[2][1])
                ax.set_zlim(boundaries[1][0], boundaries[1][1])
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_zlabel("y")
                ax.invert_zaxis()
                if colorful:
                    trj = ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1],
                                     c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    fig.colorbar(trj)
                else:
                    ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1],
                               s=1)
                ax.set_box_aspect(
                    aspect=(float(boundaries[0][0]) - float(boundaries[0][1]),
                            float(boundaries[2][0]) - float(boundaries[2][1]),
                            float(boundaries[1][0]) - float(boundaries[1][1])))
                ax.view_init(elev=20., azim=i*0.5)
                # ax.view_init(elev=20., azim=0.5)
                ax.grid(True, which='both')
            else:
                # Note: boundaries should still be permuted
                ax = fig.add_subplot(projection='3d', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[1][0], boundaries[1][1])
                ax.set_zlim(boundaries[2][0], boundaries[2][1])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.invert_zaxis()
                if colorful:
                    trj = ax.scatter(positions[i][:, 0], positions[i][:, 1], positions[i][:, 2],
                                     c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    fig.colorbar(trj)
                else:
                    ax.scatter(positions[i][:, 0], positions[i][:, 1], positions[i][:, 2],
                               s=1)
                ax.set_box_aspect(
                    aspect=(float(boundaries[0][0]) - float(boundaries[0][1]),
                            float(boundaries[1][0]) - float(boundaries[1][1]),
                            float(boundaries[2][0]) - float(boundaries[2][1])))
                ax.view_init(elev=20., azim=i * 0.5)
                # ax.view_init(elev=20., azim=0.5)
                ax.grid(True, which='both')

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=20)

    ani.save(save_path, dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {save_path}")

def get_first_and_last_frame(positions, boundaries, colorbar=False, colorful=True):

    ndim = positions.shape[-1]

    # compute vel magnitude for color bar
    if colorful:
        initial_vel = np.zeros(positions[0].shape)
        initial_vel = initial_vel.reshape((1, initial_vel.shape[0], initial_vel.shape[1]))
        vel = positions[1:] - positions[:-1]
        vel = np.concatenate((initial_vel, vel))
        vel_magnitude = np.linalg.norm(vel, axis=-1)

    first_frame = positions[0]
    last_frame = positions[-1]

    if ndim == 2:

        if colorful:
                cmap = plt.cm.viridis
                vmax = np.ndarray.flatten(vel_magnitude).max()
                vmin = np.ndarray.flatten(vel_magnitude).min()
                vel_first = vel_magnitude[0]
                vel_last = vel_magnitude[-1]

        # Plot first frame
        fig1, ax1 = plt.subplots()
        ax1.set_xlim(boundaries[0][0], boundaries[0][1])
        ax1.set_ylim(boundaries[1][0], boundaries[1][1])
        
        if colorful:
                    trj = ax1.scatter(first_frame[:, 0], first_frame[:, 1],
                                     c=vel_first, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    if colorbar:
                        fig1.colorbar(trj)
        else:
            ax1.scatter(first_frame[:, 0], first_frame[:, 1], s=1)
        
        ax1.grid(True, which='both')

        # Plot last frame
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(boundaries[0][0], boundaries[0][1])
        ax2.set_ylim(boundaries[1][0], boundaries[1][1])

        if colorful:
                    trj = ax2.scatter(last_frame[:, 0], last_frame[:, 1],
                                     c=vel_last, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    if colorbar:
                        fig2.colorbar(trj)
        else:
            ax2.scatter(last_frame[:, 0], last_frame[:, 1], s=1)

        ax2.scatter(last_frame[:, 0], last_frame[:, 1], s=1)
        ax2.grid(True, which='both')

    elif ndim == 3:

        if colorful:
                cmap = plt.cm.viridis
                vmax = np.ndarray.flatten(vel_magnitude).max()
                vmin = np.ndarray.flatten(vel_magnitude).min()
                vel_first = vel_magnitude[0]
                vel_last = vel_magnitude[-1]

        # Plot first frame
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')
        ax1.set_xlim(boundaries[0][0], boundaries[0][1])
        ax1.set_ylim(boundaries[1][0], boundaries[1][1])
        ax1.set_zlim(boundaries[2][0], boundaries[2][1])

        if colorful:
                    trj = ax1.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2],
                                     c=vel_first, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    if colorbar:
                        fig1.colorbar(trj)
        else:
            ax1.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], s=1)

        # Plot last frame
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection='3d')
        ax2.set_xlim(boundaries[0][0], boundaries[0][1])
        ax2.set_ylim(boundaries[1][0], boundaries[1][1])
        ax2.set_zlim(boundaries[2][0], boundaries[2][1])

        if colorful:
                    trj = ax2.scatter(last_frame[:, 0], last_frame[:, 1], last_frame[:, 2],
                                     c=vel_last, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    if colorbar:
                        fig2.colorbar(trj)
        else:
            ax2.scatter(last_frame[:, 0], last_frame[:, 1], last_frame[:, 2], s=1)

    return fig1, fig2

def positions_from_rollout(path):
    
    with open(path, 'rb') as f:
        rollout_data = pickle.load(f)
    
    mpm = np.concatenate([rollout_data["initial_positions"], 
                                       rollout_data["ground_truth_rollout"]], axis=0)
    gns = np.concatenate([rollout_data["initial_positions"],
                                        rollout_data["predicted_rollout"]], axis=0)
    
    return mpm, gns

def plot_comparison(trajectories, boundaries, angles, output_path):
    
    fig, axes = plt.subplots(
    4, 7,  # Change from (4, 5) to (4, 7) to introduce spacer columns
    figsize=(28, 20),  # Increase figure width to accommodate extra space
    gridspec_kw={'width_ratios': [1, 0.2, 1, 1, 0.2, 1, 1], 'wspace': 0.3}  # Add narrow empty columns
)
    
    for i, angle in enumerate(angles):
        mpm, gns = trajectories[angle]
        
        initial_vel_mpm = np.zeros(mpm[0].shape)
        initial_vel_mpm = initial_vel_mpm.reshape((1, initial_vel_mpm.shape[0], initial_vel_mpm.shape[1]))
        vel_mpm = (mpm[1:] - mpm[:-1])/0.0025
        vel_mpm = np.concatenate((initial_vel_mpm, vel_mpm))
        vel_magnitude_mpm = np.linalg.norm(vel_mpm, axis=-1)
         
        cmap = plt.cm.viridis
        vmax = np.ndarray.flatten(vel_magnitude_mpm).max()
        vmin = np.ndarray.flatten(vel_magnitude_mpm).min()
        vel_first_mpm = vel_magnitude_mpm[0]
        vel_second_mpm = vel_magnitude_mpm[149]
        vel_third_mpm =  vel_magnitude_mpm[299]

        initial_vel_gns = np.zeros(gns[0].shape)
        initial_vel_gns = initial_vel_gns.reshape((1, initial_vel_gns.shape[0], initial_vel_gns.shape[1]))
        vel_gns = (gns[1:] - gns[:-1]) / 0.0025
        vel_gns = np.concatenate((initial_vel_gns, vel_gns))
        vel_magnitude_gns = np.linalg.norm(vel_gns, axis=-1)
         
        cmap = plt.cm.viridis
        vmax = np.ndarray.flatten(vel_magnitude_gns).max()
        vmin = np.ndarray.flatten(vel_magnitude_gns).min()
        vel_second_gns = vel_magnitude_gns[149]
        vel_third_gns =  vel_magnitude_gns[299]

        first_frame_mpm = mpm[0]
        second_frame_mpm = mpm[149]
        third_frame_mpm = mpm[299]
        second_frame_gns = gns[149]
        third_frame_gns = gns[299]

        # Plot first frame
        axes[i, 0].set_xlim(boundaries[0][0], boundaries[0][1])
        axes[i, 0].set_ylim(boundaries[1][0], boundaries[1][1])

        trj = axes[i, 0].scatter(first_frame_mpm[:, 0], first_frame_mpm[:, 1],
                                     c=vel_first_mpm, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
        axes[i, 0].set_ylabel(" y (m)", fontsize=30)
        axes[i, 0].set_title("MPM & GNS", fontsize=30)
        axes[i, 0].set_xticks([0, 0.5, 1])
        axes[i, 0].set_xticklabels([0, 0.5, 1], fontsize=20)
        axes[i, 0].set_yticks([0, 0.5, 1])
        axes[i, 0].set_yticklabels([0, 0.5, 1], fontsize=20)
        
        # Plot last frame
        axes[i, 2].set_xlim(boundaries[0][0], boundaries[0][1])
        axes[i, 2].set_ylim(boundaries[1][0], boundaries[1][1])
        trj = axes[i, 2].scatter(second_frame_mpm[:, 0], second_frame_mpm[:, 1],
                                     c=vel_second_mpm, vmin=vmin, vmax=vmax, cmap=cmap, s=1)

        #axes[i, 2].set_ylabel(" y (m)", fontsize=30)
        axes[i, 2].set_title("MPM", fontsize=30)
        axes[i, 2].set_xticks([0, 0.5, 1])
        axes[i, 2].set_xticklabels([0, 0.5, 1], fontsize=20)
        axes[i, 2].set_yticks([0, 0.5, 1])
        axes[i, 2].set_yticklabels([0, 0.5, 1], fontsize=20)


        # Plot first frame
        axes[i, 3].set_xlim(boundaries[0][0], boundaries[0][1])
        axes[i, 3].set_ylim(boundaries[1][0], boundaries[1][1])

        trj = axes[i, 3].scatter(second_frame_gns[:, 0],second_frame_gns[:, 1],
                                     c=vel_second_gns, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
        #axes[i, 3].set_ylabel(" y (m)", fontsize=30)
        axes[i, 3].set_title("GNS", fontsize=30)
        axes[i, 3].set_xticks([0, 0.5, 1])
        axes[i, 3].set_xticklabels([0, 0.5, 1], fontsize=20)
        axes[i, 3].set_yticks([0, 0.5, 1])
        axes[i, 3].set_yticklabels([0, 0.5, 1], fontsize=20)
        
        # Plot last frame
        axes[i, 5].set_xlim(boundaries[0][0], boundaries[0][1])
        axes[i, 5].set_ylim(boundaries[1][0], boundaries[1][1])
        trj = axes[i, 5].scatter(third_frame_mpm[:, 0], third_frame_mpm[:, 1],
                                     c=vel_third_mpm, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
        #axes[i, 5].set_ylabel(" y (m)", fontsize=30)
        axes[i, 5].set_title("MPM", fontsize=30)
        axes[i, 5].set_xticks([0, 0.5, 1])
        axes[i, 5].set_xticklabels([0, 0.5, 1], fontsize=20)
        axes[i, 5].set_yticks([0, 0.5, 1])
        axes[i, 5].set_yticklabels([0, 0.5, 1], fontsize=20)

        # Plot first frame
        axes[i, 6].set_xlim(boundaries[0][0], boundaries[0][1])
        axes[i, 6].set_ylim(boundaries[1][0], boundaries[1][1])

        trj = axes[i, 6].scatter(third_frame_gns[:, 0],third_frame_gns[:, 1],
                                     c=vel_third_gns, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
        #axes[i, 6].set_ylabel(" y (m)", fontsize=30)
        axes[i, 6].set_title("GNS", fontsize=30)
        axes[i, 6].set_xticks([0, 0.5, 1])
        axes[i, 6].set_xticklabels([0, 0.5, 1], fontsize=20)
        axes[i, 6].set_yticks([0, 0.5, 1])
        axes[i, 6].set_yticklabels([0, 0.5, 1], fontsize=20)

        axes[i, 1].axis("off")  # Hide column 1
        axes[i, 4].axis("off")

        cbar = fig.colorbar(trj, ax=axes[i, 6])
        cbar.ax.tick_params(labelsize=20)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.7)
    fig.savefig(os.path.join(output_path, "rollout_comparison.png"), dpi=300, bbox_inches='tight', transparent=True)

def plot_rollout_comparison(angles, trajectories, boundaries, output_path):
    fig, axes = plt.subplots(4, 7, figsize=(24, 20), 
                             gridspec_kw={'width_ratios': [1, 0.1, 1, 1, 0.1, 1, 1]})
    
    for i, angle in enumerate(angles):
        print(angle)
        mpm, gns = trajectories[angle]
        
        initial_vel_mpm = np.zeros(mpm[0].shape).reshape((1, *mpm[0].shape))
        vel_mpm = np.concatenate((initial_vel_mpm, (mpm[1:] - mpm[:-1]) / 0.0025))
        vel_magnitude_mpm = np.linalg.norm(vel_mpm, axis=-1)
        
        initial_vel_gns = np.zeros(gns[0].shape).reshape((1, *gns[0].shape))
        vel_gns = np.concatenate((initial_vel_gns, (gns[1:] - gns[:-1]) / 0.0025))
        vel_magnitude_gns = np.linalg.norm(vel_gns, axis=-1)
        
        cmap = plt.cm.viridis
        vmax = max(vel_magnitude_mpm.max(), vel_magnitude_gns.max())
        vmin = min(vel_magnitude_mpm.min(), vel_magnitude_gns.min())
        
        frames_mpm = [mpm[0], mpm[149], mpm[299]]
        frames_gns = [gns[149], gns[299]]
        vel_mpm = [vel_magnitude_mpm[0], vel_magnitude_mpm[149], vel_magnitude_mpm[299]]
        vel_gns = [vel_magnitude_gns[149], vel_magnitude_gns[299]]
        
        for j, (frame, vel, title) in enumerate(zip(frames_mpm + frames_gns, 
                                                     vel_mpm + vel_gns, 
                                                     ["MPM", "MPM", "", "MPM", "GNS"])):
            if j == 1 or j == 4:
                continue  # Skip the empty space columns
            col_idx = j 
            #if j < 1 else j + 1 if j < 3 else j + 2
            
            axes[i, col_idx].set_xlim(boundaries[0][0], boundaries[0][1])
            axes[i, col_idx].set_ylim(boundaries[1][0], boundaries[1][1])
            trj = axes[i, col_idx].scatter(frame[:, 0], frame[:, 1], c=vel, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
            
            if title:
                axes[i, col_idx].set_title(title, fontsize=30)
            
            axes[i, col_idx].set_xticks([0, 0.5, 1])
            axes[i, col_idx].set_xticklabels([0, 0.5, 1], fontsize=20)
            axes[i, col_idx].set_yticks([0, 0.5, 1])
            axes[i, col_idx].set_yticklabels([0, 0.5, 1], fontsize=20)
            
            if col_idx == 0:
                axes[i, col_idx].set_ylabel("y (m)", fontsize=30)
        
        cbar = fig.colorbar(trj, ax=axes[i, 6])
        cbar.ax.tick_params(labelsize=20)
    
    plt.subplots_adjust(wspace=0.5, hspace=0.7)
    fig.savefig(os.path.join(output_path, "rollout_comparison.png"), dpi=300, bbox_inches='tight', transparent=True)
    

rollout_path = "../gns-reptile-task-encoder/plots/poster/rollouts/"

angles = [20, 25, 35, 40]
trajectories = {}
for angle in angles:
    pkl_file_path = os.path.join(rollout_path, f"{angle}_deg_0_ex0.pkl")
    trajectories[angle] = positions_from_rollout(pkl_file_path)

boundaries = [[0,1],
              [0,1]]
output_path = "../gns-reptile-task-encoder/plots/poster"

#plot_comparison(angles, trajectories, boundaries, output_path)
plot_comparison(trajectories, boundaries, angles, output_path)
'''input_base_path = "../terramechanics/datasets/npz_files/obstacles_3d/"
output_base_path = "../terramechanics/datasets/gifs/obstacles_3d/"
boundaries = [[0, 1],
              [0, 1],
              [0, 1]]

for i in [2]:
    input_path = os.path.join(input_base_path, f"30_deg_{i}.npz")
    output_path = os.path.join(output_base_path, f"30_deg_{i}.gif")
    animation_from_npz(input_path, output_path, boundaries)'''