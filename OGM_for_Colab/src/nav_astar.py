######################################################
# Perform navigation using A* algorithm as implemented 
# in algorihtms/astar.py
######################################################

# Temporary fix for ModuleNotFoundError.
import os
import sys
sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from pyrc3d.agent import Car
from pyrc3d.sensors import Lidar
from pyrc3d.simulation import Sim

from algorithms.Controller import PID
from algorithms.PathPlanning import AStar
from algorithms.Mapping.cheat import Cheat

from utilities.timings import Timings

# Declare user-specific paths to files.
ENV_PATH = "configs/env/simple_env.yaml"
CAR_PATH = "configs/car/car_config.yaml"
CAR_URDF_PATH = "configs/resources/f10_racecar/racecar_differential.urdf"

# FPS constants.
CTRL_FPS = 100  # Perform control at 100Hz

# Declare sensors.
SENSORS = [Lidar]

# For debugging
DEBUG=False

def main():
    # Init. starting position of agent.
    start, goal = (-8, -9), (7, 7.5)

    # Init. and create env.
    sim = Sim(time_step_freq=120, debug=DEBUG)
    sim.create_env(
            env_config=ENV_PATH,
            agent_pos=start,
            goal_loc=goal
        )

    # Init. and place car.
    car = Car(
        urdf_path=CAR_URDF_PATH, 
        car_config=CAR_PATH, 
        sensors=SENSORS
    )
    car.place_car(
            sim.floor, xy_coord=start
        )
    
    # Set simulation response time.
    ctrl_time = Timings(CTRL_FPS)

    # Map the environment
    res = 0.05 
    mapper = Cheat(sim, res)
    mapper.generate_grid_map()

    # Find shortest path to goal
    path_planner = AStar(
            mapper, 
            start=start,
            goal=goal
        )
    path = path_planner.plan_path()
    mapper.save_grid_map(shortest_path=path)

    # Initiate controller
    controller = PID(path=path, debug=DEBUG)

    # Get the car's state (pose): (x, y, theta)
    x, y, yaw = car.get_state(to_array=False)

    # Begin navigation.
    while True:
        # Perform car controls.
        v, s = controller.navigate(x, y, yaw)

        # Update car's state.
        if ctrl_time.update_time():
            car.act(v, s) # Perform action (drive)

            # Advance one time step in the simulation.
            sim.step()

            # Get updated states.
            x, y, yaw = car.get_state(to_array=False)
            
if __name__ == '__main__':
    main()
