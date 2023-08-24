import numpy as np
import matplotlib.pyplot as plt

class Cheat():
    """
    Implements a 'cheating' way of generating
    a map for the simulation environment.

    It uses information from the simulation env.
    to generate the map. In particular, it takes
    the list of obstacles' centers coordinate from `sim`
    and generate a grid map of occupied and unoccupied
    cells.
    """
    def __init__(
            self, 
            sim, 
            res,
            car_width_buffer=0.3
        ) -> None:
        """
        Args:
            obstacle_width_buffer: A value to mimic car's
                                   width, allowing buffer when squeezing
                                   in between obstacles. Capped at 0.3.
        """
        self.res = res

        self.world_width = sim.floor_s
        self.world_height = sim.floor_s
        self.origin = (-sim.floor_s/2.0, sim.floor_s/2.0)

        self.obs_width = sim.obs_w + min(car_width_buffer, 0.2)
        self.obstacle_centers = np.array(sim.obstacle_coordinates)

        self.goal_width = sim.goal_w
        self.goal = sim.goal_loc

    def generate_grid_map(self):
        # Create empty grid
        w = self.world_width/self.res    
        h = self.world_height/self.res
        grid_map = np.zeros((int(w), int(h)))

        # Find the indexes in grid map which
        # correspond to obstacles in world map.
        obs_grid_indexes = self.__populate_grid(
                    self.obstacle_centers, self.obs_width, self.origin, self.res
                )

        # Find the indexes in grid map which
        # correspond to goal in world map. This
        # is for viewing grid map purpose only; path finding
        # algorithm do not need this.
        goal_grid_indexes = self.__populate_grid(
                    self.goal, self.goal_width, self.origin, self.res
                )
        self.goal_grid_indexes = goal_grid_indexes

        grid_map[obs_grid_indexes[:, 0], obs_grid_indexes[:, 1]] = 1
        self.grid_map = grid_map

    def save_grid_map(
            self, 
            shortest_path=None, 
            save_path='grid_map.png'
        ):

        fig, ax = plt.subplots()
        
        # Set colormap to binary (black: obstacle, white: free)
        ext = self.world_width//2
        cmap = plt.cm.get_cmap('binary').copy()

        # Set goal to be colored green
        cmap.set_bad(color='g')

        # Make a copy of grid_map so we don't modify it.
        copy_mapped = self.grid_map
        copy_mapped[self.goal_grid_indexes[:, 0], self.goal_grid_indexes[:, 1]] = np.inf
        
        # Plot the grid map
        im = ax.imshow(
            np.flipud(copy_mapped.T),
            cmap=cmap,
            extent=[-ext, ext, -ext, ext],
        )

        # Check if plotting grid is feasible
        from matplotlib.ticker import MultipleLocator
        num_xticks = self.world_width/(self.res*self.res)
        num_yticks = self.world_height/(self.res*self.res)
        num_ticks = max(num_xticks, num_yticks)
        if num_ticks < MultipleLocator.MAXTICKS:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(self.res))

            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(self.res))

            ax.xaxis.grid(True,'minor', color='grey', linestyle = 'dotted', linewidth=0.5)
            ax.yaxis.grid(True,'minor', color='grey', linestyle = 'dotted', linewidth=0.5)
            ax.xaxis.grid(True,'major', color='grey', linewidth=1)
            ax.yaxis.grid(True,'major', color='grey', linewidth=1)
        else:
            print("Too many grid lines to plot, only resolution grid lines will be ignored.")
            print("Number of ticks: {}, MultipleLocator.MAXTICKS: {}".format(
                num_ticks, MultipleLocator.MAXTICKS
            ))
            ax.grid()
        
        if shortest_path is not None:
            plt.plot(shortest_path[:, 0], shortest_path[:, 1], 'r')

        plt.savefig(save_path)

    def world2grid(
            self,
            world_coords, 
            origin, 
            resolution, 
            totuple=False
        ):
        """
        Convert world coordinates to grid coordinates.
    
        Parameters:
        -----------
        world_coords : a tuple or ndarray
            The coordinates in world space as a tuple or array with shape (n, m).
        origin : tuple or ndarray
            The origin of the grid in world space.
        resolution : float
            The size of each grid cell in world space units.
    
        Returns:
        --------
        ndarray :
            The grid coordinates as a numpy array with shape (n, m, 2).
        """
        if isinstance(world_coords, np.ndarray):
            # `world_coords` is an ndarray of N (x, y) coordinates
            assert len(world_coords.shape) == 2 and world_coords.shape[1] == 2, \
                'np.ndarray `world_coords` must has shape (N, 2)'
            if isinstance(origin, tuple):
                origin = np.array(origin).reshape(-1, )
            if isinstance(origin, np.ndarray):
                origin = origin.reshape(-1, )
                
            x, y = world_coords[:, 0], world_coords[:, 1]
            grid_x = np.floor((x - origin[0]) / resolution).astype(int)
            grid_y = np.floor((y - origin[1]) / resolution).astype(int)
            return np.column_stack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)))
        elif isinstance(world_coords, tuple):
            # `world_coords` is a single (x, y) coordinate
            world_coords = np.array(world_coords).reshape(1, -1)
            grid_index = self.world2grid(world_coords, origin, resolution)
            if totuple:
                return tuple(grid_index.tolist()[0])
            return grid_index
        else:
            assert (isinstance(world_coords, np.ndarray) or isinstance(world_coords, tuple)), \
                'Input `world_coords` must be np.ndarray or tuple'
    
    def grid2world(
            self,
            grid_coords, 
            origin, 
            resolution,
            totuple=False
        ):
        """
        Convert grid coordinates to world coordinates.
    
        Parameters:
        -----------
        grid_coords : tuple or ndarray
            The coordinates in grid space as a tuple or array with shape (n, m).
        origin : tuple or ndarray
            The origin of the grid in world space.
        resolution : float
            The size of each grid cell in world space units.
    
        Returns:
        --------
        ndarray :
            The world coordinates as a numpy array with shape (n, m, 2).
        """
        if isinstance(grid_coords, np.ndarray):
            # `grid_coords` is an ndarray of N (x, y) coordinates
            assert len(grid_coords.shape) == 2 and grid_coords.shape[1] == 2, \
                'np.ndarray `world_coords` must has shape (N, 2)'
            if isinstance(origin, tuple):
                origin = np.array(origin).reshape(-1, )
            if isinstance(origin, np.ndarray):
                origin = origin.reshape(-1, )
    
            grid_x, grid_y = grid_coords[:, 0], grid_coords[:, 1]
            world_x = (grid_x * resolution) + origin[0]
            world_y = (grid_y * resolution) + origin[1]
            return np.column_stack((world_x.reshape(-1, 1), world_y.reshape(-1, 1)))
        elif isinstance(grid_coords, tuple):
            # `grid_coords` is a single (x, y) coordinate
            grid_coords = np.array(grid_coords).reshape(1, -1)
            world_coords = self.grid2world(grid_coords, origin, resolution)
            if totuple:
                return tuple(world_coords.tolist()[0])
            return world_coords
        else:
            assert (isinstance(grid_coords, np.ndarray) or isinstance(grid_coords, tuple)), \
                'Input `grid_coords` must be np.ndarray or tuple'

    def __populate_grid(
            self,
            centers, 
            obs_width, 
            origin, 
            res
        ):
        """
        Given a point `center` in world coordinate and a square of width `width` centered at `center`, 
        returns the grid indexes of the cells that intersect with the square.

        Parameters:
        -----------
        center : tuple
            The center of the square in world space as a tuple (x, y).
        obs_width : float
            The width of the obstacles in world space units.
        origin : tuple or ndarray
            The origin of the grid in world space.
        resolution : float
            The size of each grid cell in world space units.

        Returns:
        --------
        list of tuples :
            The grid indexes as a list of tuples with shape (n, 2).
        """
        if isinstance(centers, np.ndarray):
            # `centers` is an ndarray of N (x, y) coordinates
            assert len(centers.shape) == 2 and centers.shape[1] == 2, \
                'np.ndarray `world_coords` must has shape (N, 2)'
            if isinstance(origin, tuple):
                origin = np.array(origin).reshape(-1, )
            if isinstance(origin, np.ndarray):
                origin = origin.reshape(-1, )
            
            # Convert centers and area covered by obs_width to grid space
            width_grid = int(np.ceil(obs_width / res))
            centers_grid_idx = self.world2grid(centers, origin, res)

            # Compute the range of grid indices that intersect with the square
            xy_min = centers_grid_idx - (width_grid // 2)
            xy_max = centers_grid_idx + (width_grid // 2)
            
            x_min, y_min = xy_min[:, 0], xy_min[:, 1]
            x_max, y_max = xy_max[:, 0], xy_max[:, 1]

            # Create a list of grid indices that intersect with the square
            grid_indexes = []
            for i in range(len(centers)):
                coverage = []
                for y in range(y_min[i], y_max[i] + 1):
                    for x in range(x_min[i], x_max[i] + 1):
                        if (x, y) not in coverage:
                            # For saving compute resource.
                            coverage.append((x, y))
                grid_indexes.append(np.array(coverage))
            grid_indexes = np.array(grid_indexes)
            return grid_indexes.reshape(-1, grid_indexes.shape[-1])
        
        elif isinstance(centers, tuple):
            # `centers` is a single (x, y) coordinate
            centers = np.array(centers).reshape(1, -1)
            return self.__populate_grid(centers, obs_width, origin, res)
        else:
            assert (isinstance(centers, np.ndarray) or isinstance(centers, tuple)), \
                'Input `centers` must be np.ndarray or tuple'

