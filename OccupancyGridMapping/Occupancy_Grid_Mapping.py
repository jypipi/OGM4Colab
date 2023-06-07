# ------------------------------------------------------------------
# Binary Bayes Filter implementation of Occupancy Grid Mapping
# (OGM) | Status: COMPLETED (Version 2)
#       | Contributors: Jeffrey, Muhammad
#
# Update from previous version:
# - Make it a class (modularize) for easier application.
# - Separate it from simulation (make it another class).
# - Separate data collection from data processing to increase speed.
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
#
# Problem setting:
#     Given sensor data z and poses (states) x of the
#     robot at time step t, estimate the map:
#         p(m | z, x)
# ------------------------------------------------------------------

import OccupancyGridMapping.Simulation as Simulation
import matplotlib.pyplot as plt
import numpy as np
from math import *
from decimal import *

class OGM():
    """
    A class used to implement OGM algorithm, provide visualization,
        and output a probabilistic grid map.

    Attributes:
        sim (Simulation): Simulation of environment and robot.
        dataset (dict): Set of robot's pose and ray hit points.
        res (float): Grid map resolution.
        log_prior (float): Prior log-odd value.
        log_t (numpy 2D array): Array of log odd scores for all grids.
        gridMapSize (int): Side length (number of grids) of the grid map.
    """


    def __init__(self, res, log_prior):
        """
        Constructor of OGM to initialize occupancy grid mapping.

        Parameters:
            res (float): Grid map resolution.
            log_prior (float): Prior log-odd value.
        """
        
        self.sim, self.dataset = Simulation.main()
        self.res = res
        self.log_prior = log_prior
        self.log_t, self.gridMapSize = self.generate_grid_map()
        

    def generate_grid_map(self):
        """
        Generate a grid map as a matrix of zeros.
        Agent does not know the occupancies of
        each cell in the map: 
            m_i_j = 0 for all i, j.

        It will probabilistically updates it via
        bayesian approach.
        
        Args:
            None

        Returns:
            m: A numpy array of zeros of shape (map_size, map_size).
            gridMapSize (int): Side length (number of grids) of the grid map.

        Raises:
            None.
        """
        map_size = self.sim.sim.get_env_info()["map_size"]
        gridMapSize = int(map_size/self.res)
        m = np.zeros((gridMapSize, gridMapSize))
        return m, gridMapSize


    def mapping(self) -> np.ndarray:
        """
        Main function to operate occupancy grid mapping.

        Parameters:
            None
        
        Returns:
            probGridMap (numpy 2D array): Array representing
                the occupancy probability of each grid in the map.
        """

        for t in range(len(self.dataset)):
            # Extract data at time t
            pose, hitPoints = self.dataset[t][0], self.dataset[t][1]

            # Update log odds
            self.log_t = self.update_log_odds(hitPoints, pose)
        
        # Generate the probabilistic grid map based on the latest log odds
        probGridMap = self.getProbGridMap()

        return probGridMap


    def bresenham(self, x_start, y_start, x_end, y_end):
        """
        The function to determine the grids that a given line passes by,
            based on the Bresenham's line algorithm.

        Parameters:
            x_start (float): X coordinate of the starting point.
            y_start (float): Y coordinate of the starting point.
            x_end (float): X coordinate of the end point.
            y_end (float): Y coordinate of the end point.
        
        Returns:
            numpy 2D array: Array of the grids passed by the line,
                represented in their world coordinates.
        """

        # Normalize the grid side length to 1
        scale = Decimal(str(1.0)) / Decimal(str(self.res))
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


    def perceptedCells(self, xt, hitPoints):
        """
        The function to determine the grids within the ray cone,
            the range measurement of each ray, and
            the relative angle between the robot and each ray.

        Parameters:
            xt (array): Robot pose [x, y, theta].
            hitPoints (array): Array of the end points for all laser rays.
        
        Returns:
            rayConeGrids_World (numpy 2D array): Array of percepted grids in world coord.
            Zt (numpy 2D array): Array of range measurements for all rays.
            measPhi (numpy 2D array): Array of relative angles between robot and rays.
        """

        rayConeGrids_World = np.array([(0, 0)])
        Zt = np.zeros(hitPoints.shape[0])
        measPhi = np.zeros(hitPoints.shape[0])

        # Iterate thru each ray to collect data
        for i in range(hitPoints.shape[0]):
            point = hitPoints[i]

            # When there's no hit for a ray, determine its end point
            if np.isnan(point[0]):
                # Relative angle between robot and the ray (Range: RAY_START_ANG - RAY_END_ANG)
                theta_body = Decimal(str(self.sim.rayStartAng)) \
                    + (Decimal(str(self.sim.beta)) * Decimal(str(i)))
                # Convert to range: +- (RAY_END_ANG - RAY_START_ANG)/2 [positive: ccw]
                ray_robot_ang = Decimal(str(pi/2)) - theta_body

                x0 = Decimal(str(self.sim.Z_max)) \
                    * Decimal(str(cos(ray_robot_ang + Decimal(str(xt[2]))))) + Decimal(str(xt[0]))
                y0 = Decimal(str(self.sim.Z_max)) \
                    * Decimal(str(sin(ray_robot_ang + Decimal(str(xt[2]))))) + Decimal(str(xt[1]))
                point = (float(x0), float(y0))

            Zt[i] = sqrt((point[0]-xt[0])**2 + (point[1]-xt[1])**2)
            measPhi[i] = atan2(point[1]-xt[1], point[0]-xt[0]) - xt[2]
            
            # Determine the starting and end grid centers in world coordinates
            xt_grid = World2Grid(xt, self.gridMapSize, self.res)
            xtGridWorld = Grid2World(xt_grid, self.gridMapSize, self.res)
            end_grid = World2Grid(point, self.gridMapSize, self.res)
            endGridWorld = Grid2World(end_grid, self.gridMapSize, self.res)

            # Determine the grids passed by the ray
            ray_grids = self.bresenham(xtGridWorld[0],
                                       xtGridWorld[1],
                                       endGridWorld[0],
                                       endGridWorld[1]
                                       )
            
            rayConeGrids_World = np.concatenate((rayConeGrids_World, ray_grids),
                                                 axis=0
                                               )
        
        rayConeGrids_World = np.unique(rayConeGrids_World[1:], axis=0)

        return rayConeGrids_World, Zt, measPhi


    def inv_sensor_model(self, xt, grid_mi, Zt, Z_max, measPhi):
        """
        The function to implement the inverse sensor model to update
            the log odd score for each grid.

        Parameters:
            xt (array): Robot pose [x, y, theta].
            grid_mi (tuple): World coordinate of grid center.
            Zt (numpy 2D array): Array of range measurements for all rays.
            Z_max (float): Maximum measurement range of lidar.
            measPhi (numpy 2D array): Array of relative angles between robot and rays.
        
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
        if ((r > np.minimum(Z_max, Zt[k]+self.res/2.0)) or 
            (np.abs(phi-measPhi[k]) > self.sim.beta/2.0)):
            
            return self.log_prior

        elif ((Zt[k] < Z_max) and (np.abs(r-Zt[k]) < self.res/2.0*sqrt(2))):
            return l_occ

        elif (r < Zt[k]):
            return l_free
        

    def update_log_odds(self, hitPoints, xt):
        """
        The function to update the log odd scores for the percepted grids.

        Parameters:
            hitPoints (array): Array of the end points for all laser rays.
            xt (array): Robot pose [x, y, theta].
        
        Returns:
            log_t (numpy 2D array): Updated array of log odd scores for all grids.
        """

        rayConeGrids_World, Zt, measPhi = self.perceptedCells(xt, hitPoints)

        # Update the log odds for all cells with the perceptual field of lidar
        for grid in rayConeGrids_World:
            grid_coord = World2Grid(grid, self.gridMapSize, self.res)

            self.log_t[grid_coord[0]][grid_coord[1]] += \
                self.inv_sensor_model(xt, grid, Zt, self.sim.Z_max, measPhi) \
                - self.log_prior

        return self.log_t


    def getProbGridMap(self):
        """
        The function to obtain the probabilistic grid map, based on the latest log odds.

        Parameters:
            None
        
        Returns:
            probGridMap (numpy 2D array): Array of occupancy probabilities of all grids.
        """

        # Initialize the probabilistic grid map
        probGridMap = np.zeros((self.log_t.shape[0], self.log_t.shape[1]))

        # Convert log odds to probabilities and set the occupancy status of grids
        for i in range(self.log_t.shape[0]):
            for j in range(self.log_t.shape[1]):

                P_mi = 1 - 1/(1+exp(self.log_t[i][j]))

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


    def plotGridMap(self, gridMap):
        """
        The function to plot the probabilistic grid map.
            Black: the grid is occupied.
            White: the grid is free.
            Grey: Undetermined area.

        Parameters:
            gridMap (numpy 2D array): Array of occupancy probabilities of grids.
        
        Returns:
            None
        """

        plt.imshow(gridMap, cmap='gray', vmin = 0, vmax = 1, origin='lower')
        plt.show()


    def saveGridMap(self, probGridMap, path: str=None):
        """
        The function to save the probabilistic grid map to a text file.
        Default (path=None): save to the current directory.

        Parameters:
            probGridMap (numpy 2D array): Array of occupancy probabilities of all grids.
            path (str): Path of the directory and filename.
        
        Returns:
            text file (.txt): File saving the array of occupancy probabilities.
        """

        if path != None:
            return np.savetxt(path, probGridMap)
        
        return np.savetxt("probGridMap.txt", probGridMap)


def World2Grid(point, gridMapSize, res=0.1):
    """
    A helper method to determine the grid (in grid map coordinate)
        that the given point (in world coordinate) belongs to.
        It would also be imported and used by RRT implementation.

    Parameters:
        point (tuple): World coordinate of the point.
        gridMapSize (int): Side length (number of grids) of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        tuple: Grid map coordinate of the grid.
    """

    x, y = Decimal(str(point[0])), Decimal(str(point[1]))
    gridSize, res = Decimal(str(gridMapSize)), Decimal(str(res))

    i = int(np.floor((x + gridSize/2 * res) / res))
    j = int(np.floor((y + gridSize/2 * res) / res))

    return (i, j)
    

def Grid2World(grid, gridMapSize, res=0.1):
    """
    A helper method to convert a given grid map coordinate
        to the world coordinate of the grid center.
        It would also be imported and used by RRT implementation.

    Parameters:
        grid (tuple): Grid map coordinate.
        gridMapSize (int): Side length (number of grids) of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        tuple: Grid coordinate of the grid.
    """

    i, j = grid[0], grid[1]
        
    x = str((i * res) - (gridMapSize/2 * res) + res/2)
    y = str((j * res) - (gridMapSize/2 * res) + res/2)

    x = Decimal(x).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
    y = Decimal(y).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")

    return (float(x), float(y))


def main(resolution=0.1, log_prior=0.0, save_grid_map=False, path: str=None):
    """
    The function to run the program, plot the grid map, and save the
        grid map as a text file if necessary.

    Parameters:
        resolution (float): Grid map resolution.
        log_prior (float): Prior log-odd value.
        save_grid_map (bool): True = save to a text file, False otherwise.
        path (str): Path of the directory and filename. 
    Returns:
        None
    """

    ogm = OGM(resolution, log_prior)

    print("Data collection finished, now begin analyze...")
    
    probGridMap = ogm.mapping()

    # Plot the grid map
    ogm.plotGridMap(probGridMap)

    if save_grid_map:
        ogm.saveGridMap(probGridMap, path)


if __name__ == '__main__':
    main()