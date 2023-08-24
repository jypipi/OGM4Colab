################################################################################
################################################################################
# This is still in draft! A lof of bugs.
################################################################################
################################################################################

import scipy.stats
import numpy as np

class OGM():
    """
    Implements Occupancy Grid Mapping.

    Note: The pose data of the car is in the form of (x, y, yaw), where
    `x` and `y` are in the world's frame, wheres `yaw` is in the car's frame.
    
    This Occupancy Grid Mapping class assumes a square shaped simulation
    world for now.

    This code is optimized for speed and is taken/inspired/referenced from:

      https://gist.github.com/superjax/33151f018407244cb61402e094099c1d
    """

    def __init__(
            self,
            world_map_size,
            res,
            alpha=0.1,
            beta=np.pi/2.0,
            z_max=8
        ):
        
        self.res = float(res)

        self.grid_dim = int(world_map_size/self.res) # this is xsize and ysize

        self.log_prob_map = np.zeros((self.grid_dim, self.grid_dim))

        # Some constants.
        self.alpha = alpha
        self.beta = beta
        self.z_max = z_max

        self.grid_position_m = np.array(
                    [np.tile(np.arange(-self.grid_dim/2, self.grid_dim/2, self.res).reshape(-1, 1), (1, self.grid_dim)),
                     np.tile(np.arange(-self.grid_dim/2, self.grid_dim/2, self.res).reshape(1, -1), (self.grid_dim, 1))]
                )

        # Log-Probabilities to add or remove from the map 
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)

    def update(self, pose, z):
        dx = self.grid_position_m.copy() # A tensor of coordinates of all cells
        dx[0, :, :] -= pose[0] # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= pose[1] # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2] # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot

        # For each laser beam
        for z_i in z:
            r = z_i[0] # range measured
            b = z_i[1] # bearing measured

            # Calculate which cells are measured free or occupied, so we know which cells to update
            # Doing it this way is like a billion times faster than looping through each cell (because 
            # vectorized numpy is the only way to numpy)
            free_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (dist_to_grid < (r - self.alpha/2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (np.abs(dist_to_grid - r) <= self.alpha/2.0)

            # Adjust the cells appropriately
            self.log_prob_map[occ_mask] += self.l_occ
            self.log_prob_map[free_mask] += self.l_free
