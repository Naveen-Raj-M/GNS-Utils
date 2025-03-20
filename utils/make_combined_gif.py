import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TYPE_TO_COLOR = {
    1: "red",  # for droplet
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}

class Render:
    def __init__(self, input_dirs, output_dir):
        self.input_dirs = input_dirs
        self.trajectory = {}
        self.loss = {}
        self.num_steps = 0
        self.num_particles = 0
        self.output_dir = output_dir
        self.boundaries = []
        self.output_name = "combined_animation"

        # Load data from all directories and .pkl files
        self.pkl_files_per_dir = {}
        for input_dir in self.input_dirs:
            pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
            self.pkl_files_per_dir[input_dir] = pkl_files
            self.trajectory[input_dir] = {}
            self.loss[input_dir] = {}

            for idx, pkl_file in enumerate(pkl_files):
                file_path = os.path.join(input_dir, pkl_file)

                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                # Extract ground truth rollout from the first file
                if idx == 0:
                    ground_truth = np.concatenate(
                        (data['initial_positions'], data['ground_truth_rollout'])
                    )
                    self.trajectory[input_dir]["reality"] = ground_truth
                    self.boundaries = data["metadata"]["bounds"]
                    self.particle_type = data["particle_types"]
                    self.num_steps = ground_truth.shape[0]
                    self.num_particles = ground_truth.shape[1]

                # Use the filename without the '.pkl' extension as the key
                key_name = os.path.splitext(pkl_file)[0]
                # Append predicted rollout for this file
                appended_predicted_rollout = np.concatenate(
                    (data['initial_positions'], data['predicted_rollout'])
                )
                self.trajectory[input_dir][key_name] = appended_predicted_rollout
                self.loss[input_dir][key_name] = data.get("loss", "N/A")

    def color_mask(self):
        """
        Get color mask and corresponding colors for visualization.
        """
        color_mask = []
        for material_id, color in TYPE_TO_COLOR.items():
            mask = np.array(self.particle_type) == material_id
            if mask.any() == True:
                color_mask.append([mask, color])
        return color_mask

    def render_combined_gif_animation(self, point_size=1, timestep_stride=3):
        """
        Render a combined `.gif` animation from multiple directories and .pkl files.
        """
        # Calculate the grid layout
        num_rows = len(self.input_dirs)
        num_cols = max(len(files) +1  for files in self.pkl_files_per_dir.values())  # Max number of .pkl files in any directory

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
        axes = np.array(axes)  # Ensure axes is an array for easier indexing

        color_mask = self.color_mask()

        def animate(i):
            print(f"Render step {i}/{self.num_steps}")

            for row, input_dir in enumerate(self.input_dirs):
                trajectory = self.trajectory[input_dir]
                loss = self.loss[input_dir]
                pkl_files = self.pkl_files_per_dir[input_dir]

                for col, (key, datacase) in enumerate(trajectory.items()):
                    ax = axes[row, col]
                    ax.clear()  

                    # Set axis properties again after clearing
                    ax.set_aspect("equal")
                    ax.set_xlim([float(self.boundaries[0][0]), float(self.boundaries[0][1])])
                    ax.set_ylim([float(self.boundaries[1][0]), float(self.boundaries[1][1])])

                    for mask, color in color_mask:
                        ax.scatter(
                            datacase[i][mask, 0],
                            datacase[i][mask, 1],
                            s=point_size,
                            color=color,
                            label="" if i > 0 else key,
                        )

                    # Set the title for each subplot (title for current .pkl file)
                    if key == "reality":
                        title = "Reality"
                        subtitle = ""
                    else:
                        title = f"{key}"
                        mse = loss.get(key, "N/A")
                        subtitle = f"MSE: {mse:.2e}"

                    ax.set_title(f"{title}\n{subtitle}")
                    ax.grid(True, which="both", linestyle='-', linewidth=0.5, color='grey')

                    ax.set_xticks(np.arange(self.boundaries[0][0], self.boundaries[0][1] + 0.2, 0.2))
                    ax.set_yticks(np.arange(self.boundaries[1][0], self.boundaries[1][1] + 0.2, 0.2))

                    fig.suptitle(f"Step {i}/{self.num_steps}", fontsize=16)

        # Create animation
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=np.arange(0, self.num_steps, timestep_stride),
            interval=10,
        )

        # Save animation as GIF
        gif_path = os.path.join(self.output_dir, f"{self.output_name}.gif")
        ani.save(gif_path, dpi=100, fps=30, writer="imagemagick")
        print(f"Combined animation saved to: {gif_path}")


def main():
    input_dir_1 = "../gns-reptile/rollouts/combined_gifs/35_deg_9"
    input_dir_2 = "../gns-reptile/rollouts/combined_gifs/35_deg_10"
    output_dir = "../gns-reptile/rollouts/combined_gifs"
    input_dirs = [input_dir_1, input_dir_2]
    
    render = Render(input_dirs, output_dir)

    render.render_combined_gif_animation(
        point_size=1,
        timestep_stride=3
    )


if __name__ == "__main__":
    main()
