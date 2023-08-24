# ------------------------------------------------------------------
# Binary Bayes Filter implementation of Occupancy Grid Mapping
# (OGM) | Status: IN PROGRESS
#
# Key assumptions in OGM:
# - Robot positions (states) are known (i.e: its path is
#   known).
# - Occupancy of individual cells is independent.
# - The area that corresponds to each cell is either
#   completely free or occupied.
# - Each cell is a binary random variable that models
#   the occupancy:
#       p(m_ij = 1) = p(m_ij) := probability of cell ij is occupied
# - The world is static (obstacles are assumed not moving)
# Problem setting:
#     Given sensor data z and poses (states) x of the
#     robot at time step t, estimate the map:
#         p(m | z, x)
# ------------------------------------------------------------------

# Temporary fix for ModuleNotFoundError.
import os
import sys
sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

try:
    from pynput import keyboard
except:
    print("You do not have pynput installed! Run: pip install pynput")

from pyrc3d.agent import Car
from pyrc3d.simulation import Sim
from pyrc3d.sensors import Lidar

from utilities.timings import Timings

import matplotlib.pyplot as plt

import numpy as np

# Declare user-specific paths to files.
ENV_PATH = "configs/env/simple_env.yaml"
CAR_PATH = "configs/car/car_config.yaml"
CAR_URDF_PATH = "configs/resources/f10_racecar/racecar_differential.urdf"

# Constants.
SIMULATE_LIDAR = True

# FPS constants.
CTRL_FPS = 100 # Perform control at 100Hz
LIDAR_FPS = 30 # Simulate lidar at 30Hz
PRINT_FPS = 0.2 # Print `dist` every 5 seconds for debugging

# Declare sensors.
SENSORS = [Lidar]

def grid_to_world_coordinates(grid_coords, grid_size, res=0.1):
    i, j = grid_coords[:, 0], grid_coords[:, 1]
    x = (j * res) - (grid_size/2 * res)
    y = (i * res) - (grid_size/2 * res)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return np.hstack([x, y])

def done_navigating(key):
    try:
        if key.char == 'q':
            print("Exiting simulation")
            return False
    except AttributeError:
        pass

def generate_grid(sim, res=0.1):
    """
    Generate a grid map as a matrix of zeros.
    Agent does not know the occupancies of
    each cell in the map: 
        m_i_j = 0 for all i, j.

    It will probabilistically updates it via
    bayesian approach.
    
    Args:
        sim: The simulation environment object.
        res: The grid resolution 
                 (lower -> more grid cells).
    Returns:
        m: A numpy array of zeros of 
           shape (map_size, map_size)
    Raises:
        None.
    """
    map_size = sim.get_env_info()["map_size"]
    grid_size = int(map_size/res)
    m = np.zeros((grid_size, grid_size))
    return m, grid_size

def world_to_grid_coordinates(coords, grid_size, res=0.1):
    x, y = coords[:, 0], coords[:, 1]
    i = np.floor((y + grid_size/2 * res) / res).astype(int)
    j = np.floor((x + grid_size/2 * res) / res).astype(int)

    i = i.reshape(-1, 1)
    j = j.reshape(-1, 1)

    return np.hstack([i, j])

def ogm() -> np.ndarray:

    # Init. env.
    sim = Sim(time_step_freq=120)
    sim.create_env(env_config=ENV_PATH)

    # Init. car.
    car = Car(
        urdf_path=CAR_URDF_PATH, 
        car_config=CAR_PATH, 
        sensors=SENSORS,
        debug=True
    )
    car.place_car(sim.floor)
    
    # Set sim response time.
    ctrl_time = Timings(CTRL_FPS)
    lidar_time = Timings(LIDAR_FPS)

    print_frequency = Timings(PRINT_FPS)

    pose = None # Car's state (in this case, it's the car's pose)

    grid, grid_size = generate_grid(sim, res=0.1)

    with keyboard.Listener(on_press=done_navigating) as listener:
        # Begin navigation until q is pressed.
        while listener.running:
            # Get sensors' data.
            lidar_data = car.get_sensor_data('lidar')
            rays_data, coords = lidar_data[0], lidar_data[1]
            coords_2d = coords[:, :2]


            # Extract the rays indexes which hit the obstacles
            mask = np.where(coords_2d.any(axis=1))[0]

            grid_coords = world_to_grid_coordinates(coords_2d[mask], grid_size, res=0.1)
            grid[grid_coords[:, 0], grid_coords[:, 1]] = 1

            # diff = grid_to_world_coordinates(grid_coords, grid_size) - coords_2d[mask]
            # print("Diff:", diff)
            # print("")


            if print_frequency.update_time():
                # print('Hit points coordinates:', coords[0])
                # print("Car's pose [x, y, theta]:", pose)
                diff = grid_to_world_coordinates(grid_coords, grid_size) - coords_2d[mask]
                print("Diff:", diff)
                print("")

            if SIMULATE_LIDAR and lidar_time.update_time():
                car.simulate_sensor(
                            sensor='lidar', 
                            sensor_data=rays_data
                        )

            if ctrl_time.update_time():
                v, s, f = car.navigate()
                car.act(v, s, f)
                sim.step()
                
                pose = car.get_state()
    return grid

def plot_ogm(grid):
    plt.imshow(grid, cmap='gray', origin='lower')
    plt.show()

def main():
    grid = ogm()
    plot_ogm(grid)

if __name__ == '__main__':
    main()

