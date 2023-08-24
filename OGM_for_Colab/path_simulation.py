from math import *
import numpy as np

class PathSimulator():
    """
    Implements a manual keyboard controller
    using arrow keys
    """
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
            3: (1.95, 0.0, 0.0, 0, 1), 4: (-1.95, 0.0, 0.0, 0, -1),
            # 5: (-1.95, -1.95, 3*pi/2, 1, 1),  6: (-1.95, 1.95, pi/2, 1)
        }
        
        self.next = 1 # waypoint number

        self.time_counter = 0
        self.adjustment = 0.0

    def navigate(self, x, y, yaw):
        if self.next == 4:
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
        # else:
        #     adj_yaw = (yaw+2*pi) % (2*pi)
        #     error = heading - adj_yaw
        #     if abs(error) >= 0.05:
        #         self.turning(heading, adj_yaw, error)
        #         return self.velocity, self.steering

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
