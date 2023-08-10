import heapq
import numpy as np
from utilities import utils

class AStar():
    """
    Implements AStar algorithm using a
    grid-based approach.
    """
    def __init__(
            self,
            mapper,
            start: tuple,
            goal: tuple,
        ) -> None:
        """
        Args:
            mapper: A mapper object (`Cheat` for example).
        """
        self.mapper = mapper

        self.goal = mapper.world2grid(
                    mapper.goal,
                    mapper.origin,
                    mapper.res,
                    totuple=True
                )
        
        # We want the start path to be a little in front of the
        # car, not right underneath the base. This also fix the issue
        # of trying to make sharp turn right of the start.
        start = list(start)
        start[0] += 2.0
        self.start = mapper.world2grid(
                    tuple(start),
                    mapper.origin,
                    mapper.res,
                    totuple=True
                )

        self.grid_map = mapper.grid_map

    def heuristic(self, a, b):
        # Euclidean distance between two points
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def plan_path(self):
        """
        Returns the shortest path from start to goal using A* algorithm
        :param array: 2D numpy array representing the grid world matrix
        :param start: starting index (row, col)
        :param goal: goal index (row, col)
        :return: list of indices representing the shortest path from start to goal
        """
        # Unpack values
        goal = self.goal
        start = self.start
        grid_map = self.grid_map

        # Define possible movements from current cell to its neighbors
        movements = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        
        # Set up data structures for A* algorithm
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        heap = []
        heapq.heappush(heap, (fscore[start], start))
        
        # Loop until goal is reached or all possible paths have been explored
        print("Finding shortest path with A* Algorithm...")
        while heap:
            # Pop cell with lowest f-score off heap
            current = heapq.heappop(heap)[1]
            
            # Check if current cell is the goal
            if current == goal:
                print("######################")
                print("Shortest path found!!!")
                print("######################")
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::-1]
                path = self.mapper.grid2world(
                            np.array(path), 
                            self.mapper.origin,
                            self.mapper.res
                        )
                # Apply smoothing to remove sharp turns
                path = SavitzkyGolay(path.tolist())
                # draw path
                draw_path(path.tolist())
                return path

            # Add current cell to closed set
            close_set.add(current)
            
            # Loop through neighboring cells
            for i, j in movements:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                
                # Check if neighbor is within bounds and is not an obstacle
                if 0 <= neighbor[0] < grid_map.shape[0]:
                    if -grid_map.shape[1] <= neighbor[1] < 0:
                        if grid_map[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # Out of bounds
                        continue
                else:
                    # Out of bounds
                    continue
                
                # Check if neighbor has already been explored
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                
                # Update best path to neighbor so far
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in heap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(heap, (fscore[neighbor], neighbor))
        
        # No path found
        no_path_found()
        return None


class RRT():
    """
    Implements RRT algorithm.
    """
    def __init__(self) -> None:
        pass

############################################
# Convenient functions for all path planners.
############################################
def SavitzkyGolay(path, window_size=20, order=3):
    """
    Smooths the path using the SavitzkyGolay filter.

    Args:
        path: List of tuples representing the path.
        window_size: Window size for the filter.
        order: Polynomial order of the filter.

    Returns:
        The smoothed path as a list of tuples.
    """
    from scipy.signal import savgol_filter

    # Separate the x and y coordinates of the path
    path_x, path_y = zip(*path)

    # Apply the filter
    path_x_filtered = savgol_filter(path_x, window_size, order)
    path_y_filtered = savgol_filter(path_y, window_size, order)

    # Zip the x and y coordinates back together into a list of tuples
    smoothed_path = list(zip(path_x_filtered, path_y_filtered))

    return np.array(smoothed_path).reshape(-1, 2)

def no_path_found():
    print('#####################')
    print('### NO PATH FOUND ###')
    print('#####################')
    print(' ###             ### ')
    print('  ###           ###  ')
    print('   ###         ###   ')
    print('    ###       ###    ')
    print('     ###     ###     ')
    print('       ### ###       ')
    print('      ###   ###      ')
    print('     ###     ###     ')
    print('    ###       ###    ')
    print('   ###         ###   ')
    print('  ###           ###  ')
    print(' ###             ### ')
    print('#####################')
    print('### NO PATH FOUND ###')
    print('#####################')

def draw_path(path, z_coord=0.01, width=5, color=None):
    if color is None:
        color = utils.hex_to_rgba('#6D72C3') 

    path = [[path[0][0]-1.5, path[0][1]]] + path
    left_ptr, right_ptr = 0, 1
    while right_ptr < len(path):
        s = path[left_ptr]  + [z_coord] # 0.05 is good
        g = path[right_ptr] + [z_coord]
        utils.add_line(start=s, end=g, color=color, width=width)

        left_ptr += 1
        right_ptr += 1
