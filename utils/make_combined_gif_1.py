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
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.output_dir = input_dir
        self.output_name = "animation_combined"
        self.trajectory = {}
        self.loss = {}

        # List all .pkl files
        pkl_files = [f for f in sorted(os.listdir(self.input_dir)) if f.endswith(".pkl")]

        for idx, pkl_file in enumerate(pkl_files):
            file_path = os.path.join(self.input_dir, pkl_file)
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if idx == 0:
                # Extract metadata from the first file
                self.boundaries = data["metadata"]["bounds"]
                self.particle_type = data["particle_types"]
                ground_truth = np.concatenate(
                    (data["initial_positions"], data["ground_truth_rollout"])
                )
                self.trajectory["reality"] = ground_truth

            # Add predicted rollout for this file
            key_name = os.path.splitext(pkl_file)[0]
            predicted_rollout = np.concatenate(
                (data["initial_positions"], data["predicted_rollout"])
            )
            self.trajectory[key_name] = predicted_rollout
            self.loss[key_name] = data.get("loss", 0)

        # Trajectory information
        self.dims = self.trajectory["reality"].shape[2]
        self.num_particles = self.trajectory["reality"].shape[1]
        self.num_steps = self.trajectory["reality"].shape[0]

        # Ensure 2D data
        if self.dims != 2:
            raise ValueError("Only 2D data is supported for rendering.")

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
        Render a single `.gif` animation for all keys in `trajectory` data in 2D.

        Args:
            point_size (int): Size of particle in visualization.
            timestep_stride (int): Stride of steps to skip.

        Returns:
            gif format animation
        """
        # Increase figure size to make plots bigger
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size as needed
        ax.set_aspect("equal")
        ax.set_xlim([float(self.boundaries[0][0]), float(self.boundaries[0][1])])
        ax.set_ylim([float(self.boundaries[1][0]), float(self.boundaries[1][1])])

        # Color mask for visualization
        color_mask = self.color_mask()

        def animate(i):
            print(f"Render step {i}/{self.num_steps}")

            # Clear the figure to prevent overlapping plots
            fig.clear()

            # Create subplots for each key in self.trajectory
            num_subplots = len(self.trajectory)
            for j, (key, datacase) in enumerate(self.trajectory.items()):
                # Set the subplot
                ax = fig.add_subplot(1, num_subplots, j + 1, autoscale_on=False)
                ax.set_aspect("equal")
                ax.set_xlim([float(self.boundaries[0][0]), float(self.boundaries[0][1])])
                ax.set_ylim([float(self.boundaries[1][0]), float(self.boundaries[1][1])])

                # Plot the scatter points for each mask and color
                for mask, color in color_mask:
                    ax.scatter(
                        datacase[i][mask, 0],
                        datacase[i][mask, 1],
                        s=point_size,
                        color=color,
                        label="" if i > 0 else key,  # Add legend only in the first frame
                    )

                # Set the title for each subplot
                if key == "reality":
                    title = "Reality"
                    subtitle = ""
                else:
                    title = f"{key}"
                    mse = self.loss.get(key, "N/A")  # Use stored loss or 'N/A'
                    subtitle = f"MSE: {mse:.2e}"

                ax.set_title(f"{title}\n{subtitle}")
                ax.grid(True, which="both")

                # Set the ticks at 0.2 spacing for both axes
                ax.set_xticks(np.arange(self.boundaries[0][0], self.boundaries[0][1] + 0.2, 0.2))
                ax.set_yticks(np.arange(self.boundaries[1][0], self.boundaries[1][1] + 0.2, 0.2))

                # Ensure that grid lines are visible, solid, and grey
                ax.grid(True, which="both", linestyle='-', linewidth=0.5, color='grey')

            # Add a general title for the entire figure
            fig.suptitle(f"Step {i}/{self.num_steps}")

        # Adjust layout
        plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=2.0)
        
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
    input_dir_1 = "../gns-reptile/rollouts/few_shot/models_1_35_1"
    input_dir_2 = "../gns-reptile/rollouts/combined_gifs/40_deg_3"
    output_dir = "../gns-reptile/rollouts/few_shot/models_1_35_1"
    input_dirs = [input_dir_1, input_dir_2]
    
    render = Render(input_dir_2)

    render.render_combined_gif_animation(
        point_size=1,
        timestep_stride=3
    )


if __name__ == "__main__":
    main()
