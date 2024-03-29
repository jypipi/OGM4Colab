# ------------------------------------------------------------------
# Binary Bayes Filter implementation of Occupancy Grid Mapping
# (OGM) | Status: COMPLETED (Version 4)
#       | Contributors: Jeffrey Chen
#
# Update from previous version:
# - Enable to display a ray cone in grid map based on the user's choices.
# - Tuned refresh rate and made some modifications to increase running speed.
# - Fixed some bugs.
#
# Key assumptions in OGM:
# - Robot positions (states) are known (i.e: its path is known).
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

import matplotlib.pyplot as plt
import numpy as np
from Path_Sim import Simulation
from math import *
from decimal import *
from IPython.display import clear_output
from time import time
from copy import deepcopy
from util import World2Grid, Grid2World

class OGM():
    """
    A class used to implement OGM algorithm, provide visualization,
        and output a probabilistic grid map.

    Attributes:
        sim (Simulation): Simulation of environment and robot.
        res (float): Grid map resolution.
        log_prior (float): Prior log-odd value.
        l_occ (float): Occupancy log-odd value.
        l_free (float): Unoccupancy log-odd value.
        log_t (numpy 2D array): Array of log odd scores for all grids.
        gridMapSize (int): Side length (number of grids) of the grid map.
        realMapSize (float): Side length of the real map.
        probGridMap (numpy 2D array): Probabilistic grid map.
        plotMap (numpy 2D array): Grid map with colored ray cone for plotting.
    """


    def __init__(self, res, log_prior):
        """
        Constructor of OGM to initialize occupancy grid mapping.

        Parameters:
            res (float): Grid map resolution.
            log_prior (float): Prior log-odd value.
        """
        
        self.sim = Simulation()
        self.res = res
        self.realMapSize = self.sim.sim.get_env_info()["map_size"]
        self.gridMapSize = int(self.realMapSize/self.res)

        # Set log-odds values
        self.log_prior = log_prior
        self.l_occ = log(0.55/0.45)
        self.l_free = log(0.45/0.55)
        self.log_t = np.zeros((self.gridMapSize, self.gridMapSize))

        # Initialize the probabilistic grid map
        self.probGridMap = 0.5 * np.ones((self.log_t.shape[0], self.log_t.shape[1]))
        self.plotMap = None


    def mapping(self, dataset):
        """
        Main function to operate occupancy grid mapping. It updates the log odd
            scores and occupancy probabilities for the percepted grids.

        Parameters:
            dataset (tuple): Robot's current pose and the array of the end
                points for all laser rays.
        
        Returns:
            None
        """

        pose, hitPoints = dataset[0], dataset[1]
        rayConeGrids_World, Zt, measPhi, rayConeEnds_World = self.perceptedCells(pose, hitPoints)
        self.plotMap = deepcopy(self.probGridMap)
        self.rayConeEnds_Grid = \
            [World2Grid(i, self.realMapSize, self.gridMapSize, self.res) for i in rayConeEnds_World]

        # Update the log odds for all cells with the perceptual field of lidar
        for grid in rayConeGrids_World:
            grid_coord = World2Grid(grid, self.realMapSize, self.gridMapSize,
                                    self.res)
            i, j = grid_coord[1], grid_coord[0]

            self.log_t[i][j] += \
                self.inv_sensor_model(pose, grid, Zt, self.sim.Z_max, measPhi) \
                - self.log_prior
            
            # Convert to the occupancy probability
            P_mi = 1 - 1/(1+np.exp(self.log_t[i][j]))

            # Update the probabilistic grid map based on the latest log odds
            # When the grid is likely to be occupied
            if (P_mi > 0.5):
                self.probGridMap[i][j] = 0 # set to zero to plot in black
            # When the grid's status is likely undetermined
            elif (P_mi == 0.5):
                self.probGridMap[i][j] = 0.5 # set to 0.5 to plot in grey
            # When the grid is likely to be free
            else:
                self.probGridMap[i][j] = 1 # set to one to plot in white

            # Color the ray cone
            self.plotMap[i][j] = 0.7


    def bresenham(self, x_start, y_start, x_end, y_end):
        """
        The function to determine the grids that a given line passes by,
            based on the Bresenham's line algorithm. All coordinates are in the
            world frame.

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
            rayConeGrids_World (numpy 2D array): Array of the percepted grids
                in world coordinates.
            Zt (numpy 2D array): Array of range measurements for all rays.
            measPhi (numpy 2D array): Array of relative angles between robot
                and Lidar rays.
            rayConeEnds_World (list): Array of the rays' end points.
        """

        rayConeGrids_World = np.array([(0, 0)])
        Zt = np.zeros(hitPoints.shape[0])
        measPhi = np.zeros(hitPoints.shape[0])
        rayConeEnds_World = []

        # Iterate thru each ray to collect data
        for i in range(hitPoints.shape[0]):
            point = hitPoints[i]

            # When there's no hit for a ray, determine its end point (world)
            if np.isnan(point[0]):
                # Relative angle between robot and the ray
                # (Range: RAY_START_ANG - RAY_END_ANG)
                theta_body = Decimal(str(self.sim.rayStartAng)) \
                    + (Decimal(str(self.sim.beta)) * Decimal(str(i)))
                # Convert to range: +- (RAY_END_ANG - RAY_START_ANG)/2 [+: ccw]
                ray_robot_ang = Decimal(str(pi/2)) - theta_body

                x0 = Decimal(str(self.sim.Z_max)) \
                    * Decimal(str(cos(ray_robot_ang + Decimal(str(xt[2]))))) \
                    + Decimal(str(xt[0]))
                y0 = Decimal(str(self.sim.Z_max)) \
                    * Decimal(str(sin(ray_robot_ang + Decimal(str(xt[2]))))) \
                    + Decimal(str(xt[1]))
                point = (float(x0), float(y0))

            # Discard the invalid point
            if (abs(point[0]) >= self.realMapSize/2.0) or \
               (abs(point[1]) >= self.realMapSize/2.0):
                continue

            Zt[i] = np.sqrt((point[0]-xt[0])**2 + (point[1]-xt[1])**2)
            measPhi[i] = np.arctan2(point[1]-xt[1], point[0]-xt[0]) - xt[2]
            rayConeEnds_World.append(point)
            
            # Determine the grids passed by the ray
            ray_grids = self.bresenham(xt[0], xt[1], point[0], point[1])
            
            rayConeGrids_World = np.concatenate((rayConeGrids_World, ray_grids),
                                                 axis=0)
        
        rayConeGrids_World = np.unique(rayConeGrids_World[1:], axis=0)

        return rayConeGrids_World, Zt, measPhi, rayConeEnds_World


    def inv_sensor_model(self, xt, grid_mi, Zt, Z_max, measPhi):
        """
        The function to implement the inverse sensor model to update
            the log odd score for each grid.

        Parameters:
            xt (array): Robot pose [x, y, theta].
            grid_mi (tuple): World coordinate of grid center.
            Zt (numpy 2D array): Array of range measurements for all rays.
            Z_max (float): Maximum measurement range of lidar.
            measPhi (numpy 2D array): Array of relative angles between robot
                and Lidar rays.
        
        Returns:
            float: Log odd score update.
        """

        # Distance between robot and grid center
        r = np.sqrt((grid_mi[0]-xt[0])**2 + (grid_mi[1]-xt[1])**2)
        # Relative angle between robot and grid center
        phi = np.arctan2(grid_mi[1]-xt[1], grid_mi[0]-xt[0]) - xt[2]
        # Index of the ray that corresponds to this measurement
        k = np.argmin(abs(np.subtract(phi, measPhi)))

        # Determine the update of log odd score for this grid
        if ((r > np.minimum(Z_max, Zt[k]+self.res/2.0)) or 
            (np.abs(phi-measPhi[k]) > self.sim.beta/2.0)):
            
            return self.log_prior

        elif ((Zt[k] < Z_max) and (np.abs(r-Zt[k]) < self.res/2.0*np.sqrt(2))):
            return self.l_occ

        elif (r < Zt[k]):
            return self.l_free
        
        else:
            return 0.0


    def plotGridMap(self):
        """
        The function to plot the probabilistic grid map.
            Black: the grid is occupied.
            White: the grid is free.
            Grey: Undetermined area.

        Parameters:
            None
        
        Returns:
            None
        """

        gridMap = self.probGridMap
        plt.imshow(gridMap, cmap='gray', vmin = 0, vmax = 1, origin='lower')
        plt.title('Final Grid Map')
        plt.show()


    def saveGridMap(self, path: str=None):
        """
        The function to save the probabilistic grid map to a text file.
        Default (path=None): save to the current directory.

        Parameters:
            path (str): Path of the directory and filename.
        
        Returns:
            text file (.txt): File saving the array of occupancy probabilities.
        """

        if path != None:
            return np.savetxt(path, self.probGridMap)
        return np.savetxt("probGridMap.txt", self.probGridMap)


def main(resolution, outputImage, rayConeType, log_prior=0.0,
         save_grid_map=False, path: str=None):
    """
    The function to run the program, plot the grid map, and save the
        grid map as a text file if necessary.

    Parameters:
        resolution (float): Grid map resolution.
        outputImage (int): 1 = display PyBullet image, 2 = otherwise.
        rayConeType (int): 1 = Green Ray, 2 = Colored Grids.
        log_prior (float): Prior log-odd value.
        save_grid_map (bool): True = save to a text file, False otherwise.
        path (str): Path of the directory and filename. 
    Returns:
        None
    """

    t0 = time()
    
    ogm = OGM(resolution, log_prior)
    fig = plt.figure()

    while True:
        # Get latest data
        realMap, dataset, status = ogm.sim.collectData(outputImage)
        carPos_grid = World2Grid(dataset[0], ogm.realMapSize,
                                 ogm.gridMapSize, ogm.res)
        ogm.mapping(dataset)

        if status == -1:
            plt.clf()
            clear_output(wait=True)
            break
        
        # Display occupancy grid map and real-time real map if needed
        if realMap is not None:
            clear_output(wait=True)

            fig, ax = plt.subplots(nrows=1, ncols=2)

            ax[0].imshow(realMap)
            ax[0].set_title('Physical World')

            ax[1].imshow(ogm.probGridMap, cmap='gray', vmin=0, vmax=1,
                            origin='lower')
            for point in ogm.rayConeEnds_Grid:
                ax[1].plot([carPos_grid[0], point[0]], [carPos_grid[1], point[1]], c='g')
            ax[1].scatter(carPos_grid[0], carPos_grid[1], s=80,
                          marker='o', c='b', edgecolors='r')
            ax[1].set_title('Real-time Grid Map')

            # fig.add_subplot(1, 2, 1)
            # plt.imshow(realMap)
            # plt.title('Physical World')

            # fig.add_subplot(1, 2, 2)
            # if rayConeType == 2:
            #     plt.imshow(ogm.plotMap, cmap='gray', vmin=0, vmax=1,
            #                 origin='lower')
            # elif rayConeType == 1:
            #     plt.imshow(ogm.probGridMap, cmap='gray', vmin=0, vmax=1,
            #                 origin='lower')
            #     for point in ogm.rayConeEnds_Grid:
            #         plt.plot([carPos_grid[0], point[0]], [carPos_grid[1], point[1]], c='g')
            # plt.scatter(carPos_grid[0], carPos_grid[1], s=80,
            #             marker='o', c='b', edgecolors='r')
            # plt.title('Real-time Grid Map')
        
        #### For running on Colab
        plt.show(block=True)
        #### For running on local computer
        # plt.show(block=False)
        # plt.pause(0.001)

    print('Total run time:', floor((time()-t0)/60), 'min',
          round((time()-t0)%60, 1), 'sec.')
    
    ogm.plotGridMap()
    if save_grid_map:
        ogm.saveGridMap(path)


if __name__ == '__main__':
    res = 0.1
    input1 = input("Display PyBullet image? [1: True or 2: False]\n")
    input2 = input("Ray cone display type? [1: Green Ray, 2: Colored Grids]\n")
    outputImage = int(input1)
    rayConeType = int(input2)
    # if outputImage == 1 or outputImage == 2:
    #     if rayConeType == 1 or rayConeType == 2:
    #         main(res, outputImage, rayConeType)
    #     else:
    #         print("Invalid choice of ray cone display type.")
    # else:
    #     print("Invalid choice of whether display PyBullet image.")

    outputImage, rayConeType = True, 1
    main(res, outputImage, rayConeType)