# ------------------------------------------------------------------
# Binary Bayes Filter implementation of Occupancy Grid Mapping
# (OGM) | Status: COMPLETED (Version 1)
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
from math import *
from decimal import *
from yaml import safe_load

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

# Load car and sensor configurations
with open(CAR_PATH, 'r') as file:
    carConfig = safe_load(file)

RAY_LENGTH = carConfig['lidar_configs']['ray_len'] # length of each ray
RAY_COUNT = carConfig['lidar_configs']['num_rays'] # number of laser ray
RAY_START_ANG = carConfig['lidar_configs']['lidar_angle1'] # angle b/w robot and the 1st ray
RAY_END_ANG = carConfig['lidar_configs']['lidar_angle2'] # angle b/w robot and the last ray


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
        grid_size (int): Side length of the grid map
    Raises:
        None.
    """
    map_size = sim.get_env_info()["map_size"]
    grid_size = int(map_size/res)
    m = np.zeros((grid_size, grid_size))
    return m, grid_size


def ogm() -> np.ndarray:
    """
    Main function to operate the occupancy grid mapping.

    Parameters:
        None
    
    Returns:
        probGridMap (numpy 2D array): Array representing
            the occupancy probability of each grid in the map.
    """

    # Initial the environment
    sim = Sim(time_step_freq=120)
    sim.create_env(env_config=ENV_PATH)

    # Initial the car
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

    # Initialize car's state: (x, y, theta)
    pose = None
    
    # Set the grid map resolution
    res = 0.1

    # Initialize a blank grid map
    grid, grid_size = generate_grid(sim, res)
    
    # Get the angle b/w the 1st and last laser beam
    rayRange = (RAY_END_ANG - RAY_START_ANG) * (pi/180)
    beta = rayRange/(RAY_COUNT-1) # Angular width of each beam

    # Initialize an array of log odd scores for all grids
    log_t = np.zeros((grid_size, grid_size))
    log_prior = 0.0

    # Control the car manually to collect data
    with keyboard.Listener(on_press=done_navigating) as listener:
        # Begin navigation until q is pressed.
        while listener.running:
            # Get sensors' data.
            rays_data, coords = car.get_sensor_data('lidar')

            # Get array of hit points (x, y) in world coordinate for all rays
            hitPoints = coords[:, :2]
            
            # Update the car's pose and log odds at time t
            pose = car.get_state()
            log_t = update_log_odds(hitPoints, log_t, pose, RAY_LENGTH, beta, log_prior, grid_size, res)

            if print_frequency.update_time():
                print("Car's pose [x, y, theta]:", pose)

            if SIMULATE_LIDAR and lidar_time.update_time():
                car.simulate_sensor(
                            sensor='lidar', 
                            sensor_data=rays_data
                        )

            if ctrl_time.update_time():
                v, s, f = car.navigate()
                car.act(v, s, f)
                sim.step()
    
    # Generate the probabilistic grid map based on the latest log odds
    probGridMap = getProbGridMap(log_t)

    return probGridMap


def bresenham(x_start, y_start, x_end, y_end, res):
    """
    The function to determine the grids that a given line passes by,
        based on the Bresenham's line algorithm.

    Parameters:
        x_start (float): X coordinate of the starting point.
        y_start (float): Y coordinate of the starting point.
        x_end (float): X coordinate of the end point.
        y_end (float): Y coordinate of the end point.
        res (float): Grid map resolution.
    
    Returns:
        numpy 2D array: Array of the grids passed by the line,
            represented in their world coordinates.
    """

    # Normalize the grid side length to 1
    scale = Decimal(str(1.0)) / Decimal(str(res))
    x_start, x_end = Decimal(str(x_start))*scale, Decimal(str(x_end))*scale
    y_start, y_end = Decimal(str(y_start))*scale, Decimal(str(y_end))*scale
    
    # Check if slope > 1
    dy0 = y_end - y_start
    dx0 = x_end - x_start
    steep = abs(dy0) > abs(dx0)
    if steep:
        x_start, y_start = y_start, x_start
        x_end, y_end = y_end, x_end

    # Change direction if x_start > x_end
    if x_start > x_end:
        x_start, x_end = x_end, x_start
        y_start, y_end = y_end, y_start

    # Determine the moving direction for y
    if y_start <= y_end:
        y_step = Decimal(str(1))
    else:
        y_step = Decimal(str(-1))
    
    dx = x_end - x_start
    dy = abs(y_end - y_start)
    error = Decimal(str(0.0))
    d_error = dy / dx
    step = Decimal(str(1.0))
    yk = y_start

    perceptedGrids = []

    # Iterate over the grids from the adjusted x_start to x_end
    for xk in np.arange(x_start, x_end+step, step):
        if steep:
            new_x = yk
            new_y = xk
        else:
            new_x = xk
            new_y = yk
        
        # Scale back to the original resolution and append to the list
        perceptedGrids.append((float(new_x/scale), float(new_y/scale)))

        error += d_error
        if error >= 0.5:
            yk += y_step
            error -= step

    return np.array(perceptedGrids)


def World2Grid(point, grid_size=50, res=0.1):
    """
    A helper method to determine the grid (in grid map coordinate)
        that the given point (in world coordinate) belongs to.

    Parameters:
        point (tuple): World coordinate of the point.
        grid_size (int): Side length of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        tuple: Grid map coordinate of the grid.
    """

    x, y = Decimal(str(point[0])), Decimal(str(point[1]))
    gridSize, res = Decimal(str(grid_size)), Decimal(str(res))

    i = int(np.floor((y + gridSize/2 * res) / res))
    j = int(np.floor((x + gridSize/2 * res) / res))

    return (i, j)
    

def Grid2World(grid, grid_size=50, res=0.1):
    """
    A helper method to convert a given grid map coordinate
        to the world coordinate of the grid center.

    Parameters:
        grid (tuple): Grid map coordinate.
        grid_size (int): Side length of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        tuple: Grid coordinate of the grid.
    """

    i, j = grid[0], grid[1]
        
    x = str((j * res) - (grid_size/2 * res) + res/2)
    y = str((i * res) - (grid_size/2 * res) + res/2)

    x = Decimal(x).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
    y = Decimal(y).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")

    return (float(x), float(y))


def perceptedCells(xt, hitPoints, Z_max, beta, gridmap_size, res):
    """
    The function to determine the grids within the ray cone,
        the range measurement of each ray, and
        the relative angle between the robot and each ray.

    Parameters:
        xt (array): Robot pose [x, y, theta].
        hitPoints (array): Array of the end points for all laser rays.
        Z_max (float): Maximum measurement range of lidar.
        beta (float): Angular width of each beam.
        gridmap_size (int): Side length of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        rayConeGrids_World (numpy 2D array): Array of percepted grids in world coord.
        Zt (numpy 2D array): Array of range measurements for all rays.
        measPhi (numpy 2D array): Array of relative angles between robot and rays.
    """

    rayConeGrids_World = np.array([(0, 0)])
    Zt = np.zeros(hitPoints.shape[0])
    measPhi = np.zeros(hitPoints.shape[0])

    # Relative angle between robot and the 1st ray
    rayStartAng = RAY_START_ANG * (pi/180)

    # Iterate thru each ray to collect data
    for i in range(hitPoints.shape[0]):
        point = hitPoints[i]

        # When there's no hit for a ray, determine its end point
        if (point[0] == 0) and (point[1] == 0):
            # Relative angle between robot and the ray (Range: RAY_START_ANG - RAY_END_ANG)
            theta_body = Decimal(str(rayStartAng)) + (Decimal(str(beta)) * Decimal(str(i)))
            # Convert to range: +- (RAY_END_ANG - RAY_START_ANG)/2 [positive: ccw]
            ray_robot_ang = Decimal(str(pi/2)) - theta_body

            x0 = Decimal(str(Z_max)) * Decimal(str(cos(ray_robot_ang + Decimal(str(xt[2]))))) + Decimal(str(xt[0]))
            y0 = Decimal(str(Z_max)) * Decimal(str(sin(ray_robot_ang + Decimal(str(xt[2]))))) + Decimal(str(xt[1]))
            point = (float(x0), float(y0))

        Zt[i] = sqrt((point[0]-xt[0])**2 + (point[1]-xt[1])**2)
        measPhi[i] = atan2(point[1]-xt[1], point[0]-xt[0]) - xt[2]
        
        # Determine the starting and end grid centers in world coordinates
        xt_grid = World2Grid(xt, gridmap_size, res)
        xtGridWorld = Grid2World(xt_grid, gridmap_size, res)
        end_grid = World2Grid(point, gridmap_size, res)
        endGridWorld = Grid2World(end_grid, gridmap_size, res)

        # Determine the grids passed by the ray
        ray_grids = bresenham(xtGridWorld[0], xtGridWorld[1], endGridWorld[0], endGridWorld[1], res)
        
        rayConeGrids_World = np.concatenate((rayConeGrids_World, ray_grids), axis=0)
    
    rayConeGrids_World = np.unique(rayConeGrids_World[1:], axis=0)

    return rayConeGrids_World, Zt, measPhi


def inv_sensor_model(xt, grid_mi, Zt, Z_max, measPhi, beta, res, log_prior=0.0):
    """
    The function to implement the inverse sensor model to update
        the log odd score for each grid.

    Parameters:
        xt (array): Robot pose [x, y, theta].
        grid_mi (tuple): World coordinate of grid center.
        Zt (numpy 2D array): Array of range measurements for all rays.
        Z_max (float): Maximum measurement range of lidar.
        measPhi (numpy 2D array): Array of relative angles between robot and rays.
        beta (float): Angular width of each beam.
        res (float): Grid map resolution.
        log_prior (float): Grid map resolution.
    
    Returns:
        float: Log odd score update.
    """

    # Set log-odds values
    l_occ = log(0.55/0.45)
    l_free = log(0.45/0.55)

    # Distance between robot and grid center
    r = sqrt((grid_mi[0]-xt[0])**2 + (grid_mi[1]-xt[1])**2)
    # Relative angle between robot and grid center
    phi = atan2(grid_mi[1]-xt[1], grid_mi[0]-xt[0]) - xt[2]
    # Index of the ray that corresponds to this measurement
    k = np.argmin(abs(np.subtract(phi, measPhi)))

    # Determine the update of log odd score for this grid
    if ((r > np.minimum(Z_max, Zt[k]+res/2.0)) or (np.abs(phi-measPhi[k]) > beta/2.0)):
        return log_prior

    elif ((Zt[k] < Z_max) and (np.abs(r-Zt[k]) < res/2.0*sqrt(2))):
        return l_occ

    elif (r < Zt[k]):
        return l_free
    

def update_log_odds(hitPoints, log_t, xt, Z_max, beta, log_prior, gridmap_size, res):
    """
    The function to update the log odd scores for the percepted grids.

    Parameters:
        hitPoints (array): Array of the end points for all laser rays.
        log_t (numpy 2D array): Array of log odd scores for all grids.
        xt (array): Robot pose [x, y, theta].
        Z_max (float): Maximum measurement range of lidar.
        beta (float): Angular width of each beam.
        log_prior (float): Prior log-odd value.
        gridmap_size (int): Side length of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        log_t (numpy 2D array): Updated array of log odd scores for all grids.
    """

    rayConeGrids_World, Zt, measPhi = perceptedCells(xt, hitPoints, Z_max, beta, gridmap_size, res)

    # Update the log odds for all cells with the perceptual field of lidar
    for grid in rayConeGrids_World:
        grid_coord = World2Grid(grid, gridmap_size, res)

        log_t[grid_coord[0]][grid_coord[1]] += inv_sensor_model(xt, grid, Zt, Z_max, measPhi, beta, res, log_prior) \
                                             - log_prior

    return log_t


def getProbGridMap(log_t):
    """
    The function to obtain the probabilistic grid map, based on the latest log odds.

    Parameters:
        log_t (numpy 2D array): Array of log odd scores for all grids.
    
    Returns:
        probGridMap (numpy 2D array): Array of occupancy probabilities of all grids.
    """

    # Initialize the probabilistic grid map
    probGridMap = np.zeros((log_t.shape[0], log_t.shape[1]))

    # Convert log odds to probabilities and set the occupancy status for each grid
    for i in range(log_t.shape[0]):
        for j in range(log_t.shape[1]):

            P_mi = 1 - 1/(1+exp(log_t[i][j]))

            # When the grid is likely to be occupied
            if (P_mi > 0.5):
                probGridMap[i][j] = 0 # set to zero for plotting in black

            # When the grid's status is likely undetermined
            elif (P_mi == 0.5):
                probGridMap[i][j] = 0.5 # set to 0.5 for plotting in grey

            # When the grid is likely to be free
            else:
                probGridMap[i][j] = 1 # set to one for plotting in white

    return probGridMap


def plot_ogm(gridMap):
    """
    The function to plot the probabilistic grid map.
        Black: the grid is occupied.
        White: the grid is free.
        Grey: Undetermined area.

    Parameters:
        gridMap (numpy 2D array): Array of occupancy probabilities of all grids.
    
    Returns:
        None
    """

    plt.imshow(gridMap, cmap='gray', vmin = 0, vmax = 1, origin='lower')
    plt.show()


def main():
    """
    The function to run the program.

    Parameters:
        None
    
    Returns:
        None
    """

    probGridMap = ogm()
    plot_ogm(probGridMap)


if __name__ == '__main__':
    main()