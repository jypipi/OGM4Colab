# ------------------------------------------------------------------
# PyBullet Simulation
# | Status: COMPLETED (Version: 1)
# | Contributor: Muhammad, Jeffrey
#
# Function:
# Initialize the simulation, control the robot to collect data, and
# return the dataset.
# ------------------------------------------------------------------

try:
    from pynput import keyboard
except:
    print("You do not have pynput installed! Run: pip install pynput")

from pyrc3d.agent import Car
from pyrc3d.simulation import Sim
from pyrc3d.sensors import Lidar

from utilities.timings import Timings

from algorithms.Controller import KeyboardController

import numpy as np
from math import *
from yaml import safe_load

######### This section to load and store the simulation configuration #########

# Declare user-specific paths to files.
ENV_PATH = "configs/env/simple_env.yaml"
CAR_PATH = "configs/car/car_config.yaml"
CAR_URDF_PATH = "configs/resources/f10_racecar/racecar_differential.urdf"

# Constants.
SIMULATE_LIDAR = True

# FPS constants.
CTRL_FPS = 60 # Perform control at 100Hz
LIDAR_FPS = 60 # Simulate lidar at 30Hz
DEBUGGING_FPS = 1.0/3.0  # Print on terminal every 3 seconds for debugging
PRINT_FPS = 0.2 # Print `dist` every 5 seconds for debugging
COLLECT_DATA_FPS = 2 # Collect data frequency

# Declare sensors.
SENSORS = [Lidar]

# For debugging
DEBUG = False

# Load car and sensor configurations
with open(CAR_PATH, 'r') as file:
    carConfig = safe_load(file)

RAY_LENGTH = carConfig['lidar_configs']['ray_len'] # length of each ray
RAY_COUNT = carConfig['lidar_configs']['num_rays'] # number of laser ray
RAY_START_ANG = carConfig['lidar_configs']['lidar_angle1'] # angle b/w robot and the 1st ray
RAY_END_ANG = carConfig['lidar_configs']['lidar_angle2'] # angle b/w robot and the last ray
###############################################################################

class Simulation():
    """
    A class used to perform the simulation of environment and robot.

    Attributes:
        sim (Sim): Simulation of environment.
        beta (float): Angular width of each beam.
        rayStartAng (float): Relative angle between robot and the 1st ray.
        Z_max (float): Maximum measurement range of lidar.
    """

    def __init__(self):
        """
        Constructor of Simulation to initialize a simulation.

        Parameters:
            None
        """

        self.sim = Sim(time_step_freq=120)
        
        # Get the angle b/w the 1st and last laser beam
        rayAngleRange = (RAY_END_ANG - RAY_START_ANG) * (pi/180)
        # Angular width of each beam
        self.beta = rayAngleRange/(RAY_COUNT-1)

        # Relative angle between robot and the 1st ray
        self.rayStartAng = RAY_START_ANG * (pi/180)

        # Maximum range of each ray
        self.Z_max = RAY_LENGTH

    
    def collectData(self) -> dict:
        """
        The function to collect and store data while running the simulation.

        Parameters:
            None
        
        Returns:
            dataset (dict): Set of robot's pose and ray hit points.
        """

        # Initialize environment
        self.sim.create_env(
                env_config=ENV_PATH,
                custom_env_path='custom_map.txt'
            )
        
        # Initialize the keyboard controller
        controller = KeyboardController(CTRL_FPS)
        
        # Set simulation response time
        ctrl_time = Timings(CTRL_FPS)
        lidar_time = Timings(LIDAR_FPS)
        debugging_time = Timings(DEBUGGING_FPS)
        print_frequency = Timings(PRINT_FPS)
        collect_data_time = Timings(COLLECT_DATA_FPS)

        # Initial the car
        car = Car(
            urdf_path=CAR_URDF_PATH, 
            car_config=CAR_PATH, 
            sensors=SENSORS,
            debug=DEBUG
        )

        car.place_car(
            self.sim.floor,
            xy_coord=(0, 0)
        )

        t = 0
        dataset = {}

        with keyboard.Listener(on_press=on_press) as listener:

            # Begin navigation until q is pressed.
            while listener.running:

                # Get sensors' data: array of hit points (x, y) in world coord
                rays_data, dists, hitPoints = car.get_sensor_data(
                        sensor = 'lidar',
                        common = False
                    )
                
                (x, y, yaw) = car.get_state(to_array=False)
                
                # Store the car's pose and sensor data at time t
                if collect_data_time.update_time():
                    dataset[t] = [(x, y, yaw), hitPoints]
                    t += 1

                if print_frequency.update_time():
                    print("Car's pose [x, y, theta]:", (x, y, yaw))

                # Simulate LiDAR.
                if lidar_time.update_time():
                    car.simulate_sensor('lidar', rays_data)

                # Perform controls
                if ctrl_time.update_time():
                    v, s = controller.navigate(x, y, yaw*180/pi)

                    # Perform action (drive)
                    car.act(v, s)

                    # Advance one time step in the simulation.
                    self.sim.step()

        return dataset

def on_press(key):
    """
    A helper method to trigger the stop of simulation.
    """

    try:
        if key.char == 'q':
            # Stop listener
            return False
    except AttributeError:
        pass

def main():
    """
    The function to initialize the simulation and return the obtained dataset.

    Parameters:
        None
    
    Returns:
        sim (Simulation): Object of this simulation.
        dataset (dict): Set of robot's pose and ray hit points.
    """

    sim = Simulation()
    dataset = sim.collectData()
    return sim, dataset


if __name__ == '__main__':
    main()