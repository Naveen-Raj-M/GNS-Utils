import numpy as np
import math
import json
import math
import random
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import os
from utils.args import Config

def make_mesh(cfg: DictConfig):
    """
    Generates a mesh file for the simulation based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object with domain bounds, mesh spacing, 
                          and output paths.

    Raises:
        RuntimeError: If there is an error during mesh file creation or writing.

    Output:
        Creates a `mesh.txt` file in the specified output directory with the following format:
        - First line: Number of nodes and elements
        - Following lines: Node coordinates
        - Last lines: Connectivity of nodes forming each element
    """
    # Extract domain boundaries and spacing from configuration
    x_bounds = cfg.domain.x_bounds
    y_bounds = cfg.domain.y_bounds
    dx = cfg.domain.dx
    dy = cfg.domain.dy
    nnode_in_ele = 4  # Number of nodes per element (quadrilateral elements)

    # Calculate number of nodes and elements along each axis
    nnode_x = (x_bounds[1] - x_bounds[0]) / dx + 1
    nnode_y = (y_bounds[1] - y_bounds[0]) / dy + 1
    nele_x = nnode_x - 1
    nele_y = nnode_y - 1
    nnode = nnode_x * nnode_y  # Total number of nodes
    nele = nele_x * nele_y  # Total number of elements

    # Generate mesh node coordinates
    xs = np.arange(x_bounds[0], x_bounds[1] + dx, dx)
    ys = np.arange(y_bounds[0], y_bounds[1] + dy, dy)
    xy = []
    for y in ys:
        for x in xs:
            xy.append([x, y])
    xy = np.array(xy)

    # Generate connectivity matrix for elements
    cells = np.empty((int(nele), int(nnode_in_ele)))  # Element connectivity
    i = 0
    for ely in range(int(nele_y)):
        for elx in range(int(nele_x)):
            # Element node indices (1-based indexing is assumed in the output)
            cells[i, 0] = nnode_x * ely + elx
            cells[i, 1] = nnode_x * ely + elx + 1
            cells[i, 2] = nnode_x * (ely + 1) + elx + 1
            cells[i, 3] = nnode_x * (ely + 1) + elx
            i += 1
    cells = cells.astype(int)  # Convert indices to integers

    # Create and write the mesh file
    try:
        # Ensure output directory exists
        if not os.path.exists(cfg.output.path):
            os.makedirs(cfg.output.path)

        mesh_file = os.path.join(cfg.output.path, "mesh.txt")

        # Write the number of nodes and elements
        with open(mesh_file, "w") as f:
            f.write(f"{int(nnode)}\t{int(nele)}\n")

        # Append node coordinates
        with open(mesh_file, "a") as f:
            node_coordinates = np.array2string(
                xy, 
                separator="\t", 
                threshold=math.inf, 
                formatter={"float_kind": lambda x: f"{x:.4f}"}
            ).replace(" [", "").replace("[", "").replace("]", "")
            f.write(node_coordinates)
            f.write("\n")

        # Append element connectivity
        with open(mesh_file, "a") as f:
            element_connectivity = np.array2string(
                cells, 
                separator="\t", 
                threshold=math.inf
            ).replace(" [", "").replace("[", "").replace("]", "")
            f.write(element_connectivity)
    except Exception as e:
        raise RuntimeError(f"Failed to write mesh to file: {e}")


def create_particle_array(cfg: DictConfig) -> np.ndarray:
    """
    Generates an array of particle coordinates based on the configuration.

    Particles are distributed within the specified bounds with an optional randomness factor applied 
    to their positions.

    Args:
        cfg (DictConfig): Configuration object containing domain bounds, particle spacing, 
                          and randomness factor.

    Returns:
        np.ndarray: A 2D array where each row represents a particle's (x, y) coordinate.
    """
    # Calculate the offset and particle spacing
    offset = cfg.domain.dx / cfg.particles.nparticle_per_dir / 2
    particle_interval = cfg.domain.dx / cfg.particles.nparticle_per_dir

    # Define the bounds for particle generation
    xmin = cfg.particles.x_bounds[0] + offset
    xmax = cfg.particles.x_bounds[1] - offset
    ymin = cfg.particles.y_bounds[0] + offset
    ymax = cfg.particles.y_bounds[1] - offset

    # Generate particle coordinates along x and y directions
    xs = np.arange(xmin, xmax + offset, particle_interval)
    ys = np.arange(ymin, ymax + offset, particle_interval)

    # Create a grid of particle coordinates
    xy = [[x, y] for y in ys for x in xs]

    # Round coordinates to 5 decimal places for precision
    # xy = np.round(xy, 5)

    # Add randomness to particle positions if configured
    if cfg.particles.randomness > 0:
        random_offset = offset * cfg.particles.randomness
        xy += np.random.uniform(-random_offset, random_offset, size=np.shape(xy))

    return xy


def write_particles(particles: np.ndarray, cfg: DictConfig) -> None:
    """
    Write particle data to a file in a specified format.

    Args:
        particles (np.ndarray): 2D array containing particle coordinates.
        cfg (DictConfig): Configuration object containing output file path.

    Raises:
        Exception: If the file cannot be written.
    """
    nparticles = particles.shape[0]
    try:
        # Ensure output directory exists
        if not os.path.exists(cfg.output.path):
            os.makedirs(cfg.output.path)

        particles_file = os.path.join(cfg.output.path, "particles.txt")

        # Write the number of particles
        with open(particles_file, "w") as f:
            f.write(f"{nparticles}\n")

        # Write particle coordinates
        with open(particles_file, "a") as f:
            particle_string = np.array2string(
                particles,
                #formatter={"float_kind": lambda x: f"{x:.4f}"},
                threshold=math.inf
            ).replace(' [', '').replace('[', '').replace(']', '')
            f.write(particle_string)
    except Exception as e:
        raise RuntimeError(f"Failed to write particles to file: {e}")
    
def make_particle_stresses(particles: np.ndarray, cfg: DictConfig) -> None:
    """
    Generates a file containing the stress states of particles based on K0, density, 
    and the particle positions.

    Args:
        particles (np.ndarray): A 2D array where each row contains the (x, y) coordinates of a particle.
        cfg (DictConfig): Configuration object containing parameters for K0, density, y-bounds, and output path.

    Raises:
        RuntimeError: If there is an issue creating or writing the output file.

    Notes:
        - The stress calculations are specific to a 2D case where zz-stress is assumed to be zero.
        - Output file is named "particle-stresses.txt" and is written in the specified output path.
    """
    # Extract relevant parameters from the configuration
    k0 = cfg.particles.K0
    density = cfg.particles.density
    unit_weight = density * 9.81  # Gravitational acceleration
    y_range = cfg.particles.y_bounds

    # Initialize stress array: columns for xx, yy, zz stresses
    particle_stress = np.zeros((particles.shape[0], 3))
    particle_stress[:, 0] = k0 * (y_range[1] - particles[:, 1]) * unit_weight  # σ_xx = K0 * H * Unit_Weight
    particle_stress[:, 1] = (y_range[1] - particles[:, 1]) * unit_weight  # σ_yy = H * Unit_Weight
    particle_stress[:, 2] = 0  # σ_zz = 0 for 2D

    # Ensure output directory exists
    if not os.path.exists(cfg.output.path):
        os.makedirs(cfg.output.path)

    # Define the path to the stresses file
    stresses_file = os.path.join(cfg.output.path, "particle-stresses.txt")

    try:
        # Write the number of particles to the file
        with open(stresses_file, "w") as f:
            f.write(f"{particles.shape[0]} \n")

        # Append particle stresses to the file
        with open(stresses_file, "a") as f:
            f.write(
                np.array2string(
                    particle_stress, separator='\t', threshold=math.inf
                ).replace(' [', '').replace('[', '').replace(']', '')
            )
    except Exception as e:
        raise RuntimeError(f"Failed to write particle stresses to file: {e}")

    
def make_plot(particles: np.ndarray, cfg: DictConfig) -> None:
    """
    Generates a scatter plot of particles and saves it as 'initial_config.png'.

    Args:
        particles (np.ndarray): A 2D array where each row contains the (x, y) coordinates of a particle.
        cfg (DictConfig): Configuration object containing particle, domain, and output parameters.

    Raises:
        RuntimeError: If there is an error creating or saving the plot.

    Notes:
        - If `initial_velocity` is provided in the configuration, it displays a velocity quiver at the center of the particle domain.
        - The particle box corners are annotated with their coordinates.
        - The plot is saved to the path specified in the configuration.
    """
    # Extract bounds and metadata
    x_range = cfg.particles.x_bounds
    y_range = cfg.particles.y_bounds
    nparticles = particles.shape[0]
    nnode_in_ele = 4  # Nodes per element for particle grouping

    try:
        # Create the plot
        fig, ax = plt.subplots()
        ax.scatter(particles[:, 0], particles[:, 1], s=0.5, label="Particles")

        # Display initial velocity if available
        if cfg.particles.initial_velocity is not None:
            initial_vel = cfg.particles.initial_velocity
            # Calculate center of particle domain
            x_center = (particles[:, 0].max() - particles[:, 0].min()) / 2 + particles[:, 0].min()
            y_center = (particles[:, 1].max() - particles[:, 1].min()) / 2 + particles[:, 1].min()
            # Add quiver and velocity annotation
            ax.quiver(x_center, y_center, initial_vel[0], initial_vel[1], scale=10, color="red", label="Initial Velocity")
            ax.text(x_center, y_center, f"vel = {initial_vel}", fontsize=8)

        # Annotate the corner points of the bounding box
        ax.text(x_range[0], y_range[0], f"[{x_range[0]:.2f}, {y_range[0]:.2f}]", fontsize=8)
        ax.text(x_range[0], y_range[1], f"[{x_range[0]:.2f}, {y_range[1]:.2f}]", fontsize=8)
        ax.text(x_range[1], y_range[0], f"[{x_range[1]:.2f}, {y_range[0]:.2f}]", fontsize=8)
        ax.text(x_range[1], y_range[1], f"[{x_range[1]:.2f}, {y_range[1]:.2f}]", fontsize=8)

        # Set plot limits and aspect ratio
        ax.set_xlim(cfg.domain.x_bounds)
        ax.set_ylim(cfg.domain.y_bounds)
        ax.set_aspect('equal')

        # Add plot title and metadata
        ax.set_title(
            f"Cell size={cfg.domain.dx:.4f}x{cfg.domain.dy:.4f}, Particle/cell={nnode_in_ele**2}, nparticles={nparticles}\n"
            f"Particle coordinates: "
            f"[{x_range[0]:.2f}, {y_range[0]:.2f}], "
            f"[{x_range[0]:.2f}, {y_range[1]:.2f}], "
            f"[{x_range[1]:.2f}, {y_range[0]:.2f}], "
            f"[{x_range[1]:.2f}, {y_range[1]:.2f}]",
            fontsize=10
        )

        # Ensure the output directory exists
        if not os.path.exists(cfg.output.path):
            os.makedirs(cfg.output.path)

        # Save the plot
        plot_file = os.path.join(cfg.output.path, "initial_config.png")
        plt.savefig(plot_file)
        plt.close()
    
    except Exception as e:
        raise RuntimeError(f"Failed to write particles to file: {e}")


def write_entity_sets(particles: np.ndarray, cfg: DictConfig) -> None:
    """
    Generates entity sets for nodes and particles and writes them to a JSON file.

    Args:
        particles (np.ndarray): A 2D array where each row contains the (x, y) coordinates of a particle.
        cfg (DictConfig): Configuration object containing domain and output parameters.

    Raises:
        RuntimeError: If writing to the output file fails.

    Notes:
        - The function identifies boundary nodes and particles within the specified ranges.
        - It creates a JSON file `entity_sets.json` that contains these sets for further use.
    """
    # Extract domain and particle bounds
    x_bounds = cfg.domain.x_bounds
    y_bounds = cfg.domain.y_bounds
    dx = cfg.domain.dx
    dy = cfg.domain.dy
    x_range = cfg.particles.x_bounds
    y_range = cfg.particles.y_bounds

    # Generate grid node coordinates
    xs = np.arange(x_bounds[0], x_bounds[1] + dx, dx)
    ys = np.arange(y_bounds[0], y_bounds[1] + dy, dy)
    xy = [[x, y] for y in ys for x in xs]
    xy = np.array(xy)

    '''left_bound_node_id = []
    right_bound_node_id = []
    bottom_bound_node_id = []
    upper_bound_node_id = []
    particles_id = []

    try: 
        # Find index of nodes that match boundaries
        for i, coordinate in enumerate(xy):
            if coordinate[0] == x_bounds[0]:
                left_bound_node_id.append(i)
            if coordinate[0] == x_bounds[1]:
                right_bound_node_id.append(i)
            if coordinate[1] == y_bounds[0]:
                bottom_bound_node_id.append(i)
            if coordinate[1] == y_bounds[1]:
                upper_bound_node_id.append(i)
        for i, coordinate in enumerate(particles):
            if (x_range[0] <= coordinate[0] <= x_range[1]) \
                and (y_range[0] <= coordinate[1] <= y_range[1]):
                particles_id.append(i)

        entity_sets = {
            "node_sets": [
                {
                    "id": 0,
                    "set": bottom_bound_node_id
                },
                {
                    "id": 1,
                    "set": upper_bound_node_id
                },
                {
                    "id": 2,
                    "set": left_bound_node_id
                },
                {
                    "id": 3,
                    "set": right_bound_node_id
                }
            ],
            "particle_sets": [
                    {
                        "id": 0,
                        "set": particles_id
                    }
                ]
        }'''

    try:
        # Initialize lists for boundary node and particle indices
        x_bounds_node_id = []
        y_bounds_node_id = []
        particles_id = []

        # Find indices of nodes matching x and y boundaries
        for i, coordinate in enumerate(xy):
            if coordinate[0] == x_bounds[0] or coordinate[0] == x_bounds[1]:
                x_bounds_node_id.append(i)
            if coordinate[1] == y_bounds[0] or coordinate[1] == y_bounds[1]:
                y_bounds_node_id.append(i)

        # Find indices of particles within the particle bounds
        for i, coordinate in enumerate(particles):
            if x_range[0] <= coordinate[0] <= x_range[1] and y_range[0] <= coordinate[1] <= y_range[1]:
                particles_id.append(i)

        # Create the entity sets structure
        entity_sets = {
            "node_sets": [
                {"id": 0, "set": x_bounds_node_id},
                {"id": 1, "set": y_bounds_node_id},
            ],
            "particle_sets": [
                {"id": 0, "set": particles_id},
            ],
        }

        # Ensure the output directory exists
        if not os.path.exists(cfg.output.path):
            os.makedirs(cfg.output.path)

        # Write entity sets to a JSON file
        json_file = os.path.join(cfg.output.path, "entity_sets.json")
        with open(json_file, "w") as f:
            json.dump(entity_sets, f, indent=2)

    except Exception as e:
        raise RuntimeError(f"Failed to write entity sets to file: {e}")



def create_files(base_data: dict, cfg: DictConfig) -> None:
    """
    Generates multiple simulation input files and associated data for a range of friction angles.

    Args:
        base_data (dict): Base JSON data containing simulation input settings.
        cfg (DictConfig): Configuration object containing ranges, increments, and other parameters.

    Raises:
        RuntimeError: If any of the file creation steps fail.

    Notes:
        - Generates input files for multiple friction angles (`start_phi` to `end_phi`) with a specified increment.
        - Creates particle data, meshes, particle stresses, plots, and entity sets for each configuration.
        - Each configuration is stored in a uniquely named directory.
    """
    base_output = cfg.output.path

    # Loop over the range of angles (from `start_phi` to `end_phi` with step `increment_phi`)
    for angle in range(cfg.mpm_inputs.start_phi, cfg.mpm_inputs.end_phi + 1, cfg.mpm_inputs.increment_phi):
       
        # Generate `n_files_per_phi` files for each angle 
        for i in range(cfg.mpm_inputs.n_files_per_phi):
            
            # Create a new copy of the base data to avoid modifying the original data
            new_data = base_data.copy()
            cfg.output.path = os.path.join(base_output, f"{angle}_deg_{i}")

            # Update friction and residual friction angles in the material properties
            new_data["materials"][2]["friction"] = angle
            new_data["materials"][2]["residual_friction"] = angle

            # Randomize x-bounds for particle generation if `random_x_bounds` is enabled
            if cfg.particles.random_x_bounds:
                x_min = random.uniform(0.0, 0.7)
                cfg.particles.x_bounds = [x_min, x_min + 0.3]

            # Generate particle array
            particles = create_particle_array(cfg)

            # Write particle data to a file
            write_particles(particles, cfg)

            # Generate mesh and save it
            make_mesh(cfg)

            # Generate particle stresses if K0 is provided
            if cfg.particles.K0 is not None:
                make_particle_stresses(particles, cfg)
            else:
                print("K0 not provided. Skipping particle stresses generation.")

            # Generate a plot of the initial particle configuration
            make_plot(particles, cfg)

            # Write entity sets (node and particle groups) to a JSON file
            write_entity_sets(particles, cfg)

            # Create the output path for the modified input JSON file
            filepath = os.path.join(cfg.output.path, "mpm_input.json")

            # Write the modified JSON data to the output directory
            with open(filepath, 'w') as outfile:
                json.dump(new_data, outfile, indent=4)

            # Log the file creation
            print(f"Created file: {filepath}")



@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    """
    Main function to generate particle data and write it to a file.

    Args:
        cfg (Config): Configuration object containing parameters for particle generation
                      and output settings.
    """
    random.seed(404)

    # Load the original input data file (mpm_inputs.json or similar)   
    with open(cfg.mpm_inputs.json_file, 'r') as file:
        data = json.load(file)

    # Create MPM-input files with varying angles and cube properties
    create_files(data, cfg)
    

if __name__ == "__main__":
    main()