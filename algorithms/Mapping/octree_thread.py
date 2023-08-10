# ---------------------------------
# Implementation of Octree Mapping
# ---------------------------------

import os
import sys
import open3d as o3d
import threading

sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)

try:
    from pynput import keyboard
except:
    print("You do not have pynput installed! Run: pip install pynput")

from pyrc3d.agent import Car
from pyrc3d.simulation import Sim
from pyrc3d.sensors import Camera
from algorithms.Controller import KeyboardController
from utilities.timings import Timings

import matplotlib.pyplot as plt

import numpy as np

# Declare user-specific paths to files.
ENV_PATH = "../../src/configs/env/simple_env.yaml"
CAR_PATH = "../../src/configs/car/car_config.yaml"
CAR_URDF_PATH = "../../src/configs/resources/f10_racecar/racecar_differential.urdf"

# Constants.
SIMULATE_CAMERA = False

# FPS constants.
CTRL_FPS = 100 # Perform control at 100Hz
CAMERA_FPS = 30 # Simulate lidar at 30Hz
PRINT_FPS = 0.2 # Print `dist` every 5 seconds for debugging

# Declare sensors.
SENSORS = [Camera]

camera_data = None
pcd = None

def done_navigating(key):
    try:
        if key.char == 'q':
            print("Exiting simulation")
            return False
    except AttributeError:
        pass

def collect_point_clouds(car):
    global camera_data, pcd
    while True:
        # Get sensors' data.
        camera_data, pcd = car.get_sensor_data('camera')

def octree_mapping():
    
    """
    Implement octree mapping using Point Cloud data collected from RGBD camera
    using the RC car's relative pose in the environment.
    """
    global camera_data, pcd
    
    # Initialize simulation.
    sim = Sim(time_step_freq=120, debug=True)
    sim.create_env(env_config=ENV_PATH)

    # Init. car.
    car = Car(
        urdf_path=CAR_URDF_PATH, 
        car_config=CAR_PATH, 
        sensors=SENSORS,
        debug=True
    )
    car.place_car(sim.floor)

    # Initialize a controller
    controller = KeyboardController(CTRL_FPS)
    
    # Set sim response time.
    ctrl_time = Timings(CTRL_FPS)
    camera_time = Timings(CAMERA_FPS)

    print_frequency = Timings(PRINT_FPS)

    pose = None # Car's state (in this case, it's the car's pose)

    # Initialize octree.
    octree = o3d.geometry.Octree(max_depth=8)

    capturing_thread = threading.Thread(target=collect_point_clouds, args=(car,))

    with keyboard.Listener(on_press=done_navigating) as listener:
        # Start the capturing thread
        capturing_thread.start()

        # Begin navigation until q is pressed.
        while listener.running:

            # Get the car's current pose
            pose = car.get_state(to_array=False,radian=False)
            x, y, yaw = pose

            if pose is not None and pcd is not None:
                # Transform the point cloud to the car's pose.
                # Convert pose to a 4x4 transformation matrix.
                yaw = pose[2]
                T = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0, pose[0]],
                    [np.sin(yaw), np.cos(yaw), 0, pose[1]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                pcd.transform(T)

                # Insert the point cloud into the octree.
                octree.convert_from_point_cloud(pcd, size_expand=0.01)
           
            if print_frequency.update_time():
                print("Pose: ", pose)

            if SIMULATE_CAMERA and camera_time.update_time():
                car.simulate_sensor(
                            sensor='camera', 
                            sensor_data=camera_data
                        )

            if ctrl_time.update_time():

                v, s = controller.navigate(x,y,yaw)
                print(v,s)
                car.act(v,s) # Perform action

                # Older code
                # v, s, f = car.navigate()
                # car.act(v, s, f)
                
                sim.step() # Advance one time step in the sim
    
    # Stop the capturing thread
    capturing_thread.join()

    return octree

if __name__ == "__main__":
    octree = octree_mapping()
    o3d.visualization.draw_geometries([octree])