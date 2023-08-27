# ------------------------------------------------------------------
# PyBullet Simulation
#
# Function:
# Initialize the simulation, control the robot to collect data, and
# return the dataset.
# ------------------------------------------------------------------

# import sys
# sys.path.insert(0, '/content/test')
# !cd /

# from google.colab import output
# %matplotlib inline

from pyrc3d.agent import Car
from pyrc3d.simulation import Sim
from pyrc3d.sensors import Lidar
from utilities.timings import Timings

# from Colab_branch.pyrc3d.simulation import Sim
# from Colab_branch.pyrc3d.agent import Car
# from Colab_branch.pyrc3d.sensors import Lidar
# from Colab_branch.utilities.timings import Timings

import numpy as np
from math import *
import matplotlib.pyplot as plt
from IPython.display import clear_output

######### This section to load and store the simulation configuration #########

# Declare user-specific paths to files.
ENV_PATH = "configs/env/simple_env.yaml"
CAR_PATH = "configs/car/car_config.yaml"
CAR_URDF_PATH = "configs/resources/f10_racecar/racecar_differential.urdf"

# ENV_PATH = "/content/test/Colab_branch/configs/env/simple_env.yaml"
# CAR_PATH = "/content/test/Colab_branch/configs/car/car_config.yaml"
# CAR_URDF_PATH = "/content/test/Colab_branch/configs/resources/f10_racecar/racecar_differential.urdf"

# Constants.
SIMULATE_LIDAR = True

# FPS constants.
PATH_SIM_FPS = 100 # Perform control at 90Hz
LIDAR_FPS = 60 # Simulate lidar at 30Hz
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

        self.sim = Sim(time_step_freq=120, debug=True)

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
        path_sim = PathSimulator(car, PATH_SIM_FPS)

        t = 0
        dataset = {}

        while True:
            # try:
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
                vel, steering = path_sim.navigate(x, y, yaw)

                if vel == float('inf'):
                    break

                # Perform action (drive)
                car.act(vel, steering)

                # Advance one time step in the simulation.
                self.sim.step()
                self.sim.image_env()
            # except KeyboardInterrupt:
            #     self.sim.kill_env()

        self.sim.kill_env()

        return dataset

class PathSimulator():
    def __init__(
            self,
            car,
            sim_fps,
            acceleration=10.0,
            max_velocity=30.0,
            steering_angle_per_sec=90
        ):

        self.sim_fps = sim_fps
        self.delta_time = 1/sim_fps
        self.acceleration = acceleration  # 10 units per second^2
        self.max_velocity = max_velocity  # Maximum velocity of 50 units per second
        self.steering_angle_per_sec = steering_angle_per_sec  # Steering angle changes by 90 degrees per second

        self.velocity = 0
        self.steering = 0

        self.dist2next = 0

        self.car = car

        self.move = 0 # 1: forward, -1: backward, 0: stop
        self.ifTurn = False

        # (x, y, heading, turn)
        self.waypoints = {
            1: (1.95, 0.0, 0.0, 0, 1), 2: (-1.95, 0.0, 0.0, 0, -1),
            # 3: (1.95, 0.0, 0.0, 0, 1), 4: (-1.95, 0.0, 0.0, 0, -1),
            # 5: (-1.95, -1.95, 3*pi/2, 1, 1),  6: (-1.95, 1.95, pi/2, 1)
        }

        self.next = 1 # waypoint number

        self.time_counter = 0
        self.adjustment = 0.0

    def navigate(self, x, y, yaw):
        if self.next == 3:
            print('Reached end point.')
            return float('inf'), float('inf')

        next_x, next_y, heading, turn, move = self.waypoints[self.next]
        self.dist2next = np.linalg.norm(
            np.array((next_x, next_y))-np.array((x, y)))

        # Turn
        if self.ifTurn == False:
            if turn == 0:
                self.ifTurn = True
                return 0.0, 0.0

            adj_time_frames = round(90/45 * self.sim_fps) # frames
            self.time_counter += 1
            if self.time_counter == adj_time_frames:
                print('Turn finished')
                self.ifTurn = True
                return 0.0, 0.0

            if turn == -1: # left
                self.steering = -0.75
            elif turn == 1: # right
                self.steering = 0.75

            return 30, self.steering

        # Move
        if self.dist2next >= 0.1:
            self.moving()
            if move == -1:
                self.velocity = -self.velocity
            return self.velocity, self.steering
        else:
            self.next += 1
            print('Move to next point.')
            self.time_counter = 0
            self.ifTurn = False
            self.velocity, self.steering = 0.0, 0.0
            return self.velocity, self.steering

    def findAdj(self, error):
        if abs(error) <= (2*pi - abs(error)):
            # adjust error
            adjustment = error
        else:
            # adjust 2*pi - abs(error)
            adjustment = (2*pi - abs(error))
            if error > 0:
                adjustment *= -1
        return adjustment

    def moving(self):
        if self.dist2next > 0.6:
            self.velocity, self.steering = self.max_velocity, 0.0
        elif self.dist2next > 0.3:
            self.velocity, self.steering = 15.0, 0.0
        elif self.dist2next > 0.2:
            self.velocity, self.steering = 8.0, 0.0
        else:
            self.velocity, self.steering = 5.0, 0.0

def main():
    """
    The function to initialize the simulation and return the obtained dataset.
    """

    sim = Simulation()
    dataset = sim.collectData()
    return sim, dataset

if __name__ == '__main__':
    main()