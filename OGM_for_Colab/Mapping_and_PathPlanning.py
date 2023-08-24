# ------------------------------------------------------------------
# Mapping and Path Planning Integration
# | Status: COMPLETED (Version: 1)
# | Contributors: Jeffrey
#
# Function:
# Do all by one click: run the simulation, do occupancy grid mapping, and
# obtain a path from start to goal via RRT.
# ------------------------------------------------------------------

import OccupancyGridMapping.Occupancy_Grid_Mapping as OGM
import RRT.RRT as RRT

def main(start, goal):
    """
    The function to run the simulation, do occupancy grid mapping, and
        obtain a path from start to goal via RRT.

    Parameters:
        start (tuple): Start point in world coordinate.
        goal (tuple): Goal in world coordinate.

    Returns:
        None
    """

    # Set parameters for OGM
    resolution, log_prior, save_grid_map = 0.2, 0.0, True
    filename = "GridMapDemo.txt"

    # Mapping
    OGM.main(resolution, log_prior, save_grid_map, filename)

    # Set parameters for RRT
    epsilon, robotRadius, worldMapSize = 0.8, 0.25, 10

    # Generate path
    RRT.main(filename, start, goal, epsilon, robotRadius, worldMapSize, resolution)

if __name__ == '__main__':
    start, goal = (0.0, 0.0), (4.0, 4.0)
    main(start, goal)
    