# ------------------------------------------------------------------
# PyBullet Simulation
# | Status: COMPLETED (Version: 1)
# | Contributor: Muhammad, Jeffrey
#
# Function:
# Initialize the simulation, control the robot to collect data, and
# return the dataset.
# ------------------------------------------------------------------

import sys
sys.path.insert(0, '/content/test')
!cd /
from google.colab import output

from OGM_for_Colab.pyrc3d.agent import Car
from OGM_for_Colab.pyrc3d.simulation import Sim
from OGM_for_Colab.pyrc3d.sensors import Lidar

from OGM_for_Colab.utilities.timings import Timings

from OGM_for_Colab.path_simulation import PathSimulator

import numpy as np
from math import *
from yaml import safe_load

%matplotlib inline

######### This section to load and store the simulation configuration #########

# Declare user-specific paths to files.
ENV_PATH = "/content/test/OGM_for_Colab/configs/env/simple_env.yaml"
CAR_PATH = "/content/test/OGM_for_Colab/configs/car/car_config.yaml"
CAR_URDF_PATH = "/content/test/OGM_for_Colab/configs/resources/f10_racecar/racecar_differential.urdf"

# Constants.
SIMULATE_LIDAR = True

# FPS constants.
PATH_SIM_FPS = 60 # Perform control at 90Hz
LIDAR_FPS = 60 # Simulate lidar at 30Hz
# DEBUGGING_FPS = 1.0/3.0  # Print on terminal every 3 seconds for debugging
PRINT_FPS = 0.33 # Print `dist` every 5 seconds for debugging
COLLECT_DATA_FPS = 2 # Collect data frequency

# Declare sensors.
SENSORS = [Lidar]

# Load car and sensor configurations
RAY_LENGTH = 2.0 # length of each ray
RAY_COUNT = 50 # number of laser ray
RAY_START_ANG = 45 # angle b/w robot and the 1st ray
RAY_END_ANG = 135 # angle b/w robot and the last ray
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
                GUI=False
            )
        
        # Set simulation response time
        path_sim_time = Timings(PATH_SIM_FPS)
        lidar_time = Timings(LIDAR_FPS)
        # debugging_time = Timings(DEBUGGING_FPS)
        print_frequency = Timings(PRINT_FPS)
        collect_data_time = Timings(COLLECT_DATA_FPS)

        # Initial the car
        car = Car(
            urdf_path=CAR_URDF_PATH, 
            car_config=CAR_PATH, 
            sensors=SENSORS
        )

        car.place_car(
            self.sim.floor,
            xy_coord=(0.0, 0.0)
        )

        # Initialize path simulator
        # path_sim = PathSimulator(car, PATH_SIM_FPS)

        t = 0
        dataset = {}

        # Begin navigation until q is pressed.
        while True:

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

            ########################################### Perform controls
            if path_sim_time.update_time():
                vel, steering = 10.0, 0.0 #path_sim.navigate(x, y, yaw)

                if vel == float('inf'):
                    break

                # Perform action (drive)
                car.act(vel, steering)

                # Advance one time step in the simulation.
                self.sim.step()
                # self.sim.image_env()

        return dataset

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