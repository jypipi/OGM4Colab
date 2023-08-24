######################################################
# This script is used to collect data for
# Probabilistic Occupancy Grid Mapping.
######################################################

# Temporary fix for ModuleNotFoundError.
import os
import sys
sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import numpy as np
from pynput import keyboard

from pyrc3d.agent import Car
from pyrc3d.sensors import Lidar
from pyrc3d.simulation import Sim

from algorithms.Controller import KeyboardController
from algorithms.Mapping.ogm import OGM

from utilities.timings import Timings

# For mapping
from tqdm import tqdm
import matplotlib.pyplot as plt

# Declare user-specific paths to files.
ENV_PATH = "configs/env/simple_env.yaml"
CAR_PATH = "configs/car/car_config.yaml"
CAR_URDF_PATH = "configs/resources/f10_racecar/racecar_differential.urdf"

# FPS constants.
CTRL_FPS = 60            # Perform control at 100Hz
LIDAR_FPS = 60           # Simulate LiDAR at 60Hz
DEBUGGING_FPS = 1.0/3.0  # Print on terminal every 3 seconds
COLLECT_DATA_FPS = 15    

# Declare sensors.
SENSORS = [Lidar]

# For debugging
DEBUG = False

# For exiting while loop after finish recording data.
def on_press(key):
    try:
        if key.char == 'q':
            # Stop listener
            return False
    except AttributeError:
        pass

def main():
    # Init. starting position of agent and goal.
    start, goal = (-4, -4), (4, 4)

    if not (os.path.exists('data/poses.npy') and \
            os.path.exists('data/meas.npy')):

        # Init. env.
        sim = Sim(time_step_freq=120, debug=DEBUG)
        # Create and launch env.
        sim.create_env(
                env_config=ENV_PATH,
                agent_pos=start,
                custom_env_path='custom_map.txt',
                goal_loc=goal
            )

        # Init. car.
        car = Car(
            urdf_path=CAR_URDF_PATH, 
            car_config=CAR_PATH, 
            sensors=SENSORS,
            debug=DEBUG
        )
        # Place car.
        car.place_car(
                sim.floor, xy_coord=start
            )

        # Init. a controller.
        controller = KeyboardController(CTRL_FPS)
        
        # Set simulation response time.
        ctrl_time = Timings(CTRL_FPS)
        lidar_time = Timings(LIDAR_FPS)
        debugging_time = Timings(DEBUGGING_FPS)
        collect_data_time = Timings(COLLECT_DATA_FPS)

        # For recoding data
        poses = []
        bearings = []
        dists = []

        # Begin navigation.
        with keyboard.Listener(on_press=on_press) as listener:
            while listener.running:
                # Get the car's state (pose): (x, y, theta)
                x, y, yaw = car.get_state(to_array=False, radian=False)

                # Get LiDAR data
                lidar_data = car.get_sensor_data('lidar')
                rays_data, dist, bearing = lidar_data

                # Record lidar data, dists and bearings, and pose data.
                # every one second (bro otherwise too many data - get segfault)
                if collect_data_time.update_time():
                    poses.append([x, y, yaw])
                    bearings.append(np.radians(bearing))
                    dists.append(dist)

                # Simulate LiDAR.
                if lidar_time.update_time():
                    car.simulate_sensor('lidar', rays_data)

                # For debugging.
                if debugging_time.update_time():
                    # Print stuffs here to debug
                    pass

                # Perform controls.
                if ctrl_time.update_time():
                    v, s = controller.navigate(x, y, yaw)
                    car.act(v, s) # Perform action (drive)
                    print(v,s)

                    # Advance one time step in the simulation.
                    sim.step()
        
        # Convert data to numpy arrays
        poses = np.array(poses).reshape(3, -1)
        dists = np.array(dists).reshape(
                    car.sensors['lidar'].NUM_RAYS, -1
                )
        bearings = np.array(bearings).reshape(
                    car.sensors['lidar'].NUM_RAYS, -1
                )
        meas = np.array([dists, bearings])

        # Save them to a file for book-keeping.
        np.save('data/poses.npy', poses)
        np.save('data/meas.npy', meas)

    else:
        print("############################################")
        print("Found existing data, loading them right now!")
        print("############################################")
        poses = np.load('data/poses.npy')
        meas = np.load('data/meas.npy')

    # Setup Occupancy Grid Map settings.
    res = 1
    world_map_size = 10.0 # From config file
    max_ray_range = 8 # From config file

    # Build map.
    ogm = OGM(world_map_size, res, z_max=max_ray_range)

    plt.ion() # live plot
    plt.figure(1)
    for i in range(poses.shape[1]):
    # for i in tqdm(range(poses.shape[1])):
        ogm.update(poses[:, i], meas[:, :, i].T)

        # Real-Time Plotting 
        # (comment out these next lines to make it run super fast, matplotlib is painfully slow)
        plt.clf()
        pose = poses[:,i]
        circle = plt.Circle((pose[1], pose[0]), radius=1.0, fc='r')
        plt.gca().add_patch(circle)
        arrow = pose[0:2] + np.array([3.5, 0]).dot(np.array(
                [[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]]
            ))
        plt.plot([pose[1], arrow[1]], [pose[0], arrow[0]])
        plt.imshow(1.0 - 1./(1.+np.exp(ogm.log_prob_map)), 'Greys')
        plt.pause(0.005)
    plt.ioff()
    plt.clf()
    plt.imshow(1.0 - 1./(1.+np.exp(ogm.log_prob_map)), 'Greys') # This is probability
    plt.imshow(ogm.log_prob_map, 'Greys') # log probabilities (looks really cool)
    plt.show()

if __name__ == '__main__':
    main()
