from math import *
import numpy as np
import PID_controller as pid
from pyrc3d.agent import Car
# from pynput import keyboard

class PathSimulator():
    """
    Implements a manual keyboard controller
    using arrow keys
    """
    def __init__(
            self,
            car:Car,
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

        self.pid = pid.PID(Kp=2, Ki=2, Kd=0.001)

        self.move = 0 # 1: forward, -1: backward, 0: stop
        # self.turn = 0 # -1: left,    1: right,    0: no turn
        self.ifTurn = False

        # (x, y, heading, turn)
        # self.waypoints = {
        #     1: (1.95, 0.0, 0.0, 0),    2: (1.95, -1.95, 3*pi/2, 1), 3: (-1.95, -1.95, pi, 1),
        #     4: (-1.95, 1.95, pi/2, 1), 5: (1.95, 1.95, 0.0, 1),    6: (1.95, 0.0, 3*pi/2, 1),
        #     7: (-1.95, 0.0, pi, 1),   8: (-1.95, -1.95, 3*pi/2, -1), 9: (1.95, -1.95, 0.0, -1),
        #     10: (1.95, 1.95, pi/2, -1), 11: (-1.95, 1.95, pi, -1),     12: (-1.95, 0.0, 3*pi/2, -1),
        #     13: (0.0, 0.0, 0.0, -1),  14: (0.0, -1.95, 3*pi/2, 1), 15: (1.95, -1.95, 0.0, -1),
        #     16: (1.95, 1.95, pi/2, -1), 17: (0.0, 1.95, pi, -1),      18: (0.0, 0.0, 3*pi/2, -1)
        # }

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
        # 想法：暂时先让车子前后来回移动收集数据，可以的话转一次弯再来回移动，
        #      然后去整合进去colab page里为明天做准备，有时间的话再去fix转弯控制
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

    def turning(self, heading, yaw, error):
        # self.time_counter = 0
        # if self.time_counter % 120 == 0:
        #     self.adjustment = self.pid.adjust(heading, yaw, self.delta_time*120) # rad
        #     # self.adjustment = self.findAdj(error)
        #     # print(heading, yaw, self.adjustment)
        # self.time_counter += 1
        # adjustment = self.adjustment

        # if abs(error) <= (2*pi - abs(error)):
        #     # adjust error
        #     adjustment = error
        # else:
        #     # adjust 2*pi - abs(error)
        #     adjustment = (2*pi - abs(error))
        #     if error > 0:
        #         adjustment *= -1

        self.velocity = 10.0
        adjustment = self.pid.adjust(heading, yaw, self.delta_time)

        if adjustment >= 0.0:
            self.steering = max(adjustment/self.delta_time, -1.0)
        elif adjustment < 0.0:
            self.steering = min(adjustment/self.delta_time, 1.0)
        else:
            self.velocity, self.steering = 0.0, 0.0
            print("Error: check adjustment -> ", str(adjustment))

    def moving(self):
        if self.dist2next > 0.6:
            self.velocity, self.steering = self.max_velocity, 0.0
        elif self.dist2next > 0.3:
            self.velocity, self.steering = 15.0, 0.0
        elif self.dist2next > 0.2:
            self.velocity, self.steering = 8.0, 0.0
        else:
            self.velocity, self.steering = 5.0, 0.0






    # def navigate(self, x, y, yaw):
    #     """
    #     x, y, yaw are added as parameters for API consistency
    #     eventhough this controller does no need these values.
    #     """
    #     return 10.0, -1.0
    #     return 0.0, 0.0
    #     dist = np.linalg.norm(np.array(self.next)-np.array((x,y)))
    #     if dist <= 0.5:
    #         return self.stop()
    #         self.vel_cmd.discard('forward')
    #         self.vel_cmd.add('backward')

    #     # Calculate steering
    #     if 'right' in self.vel_cmd:
    #         self.steering += self.steering_angle_per_sec / self.sim_fps
    #         self.steering = min(self.steering, 1.0)
    #     elif 'left' in self.vel_cmd:
    #         self.steering -= self.steering_angle_per_sec / self.sim_fps
    #         self.steering = max(self.steering, -1.0)
    #     else:
    #         self.steering = 0

    #     # Calculate velocity
    #     if 'forward' in self.vel_cmd:
    #         self.velocity += self.acceleration / self.sim_fps
    #         self.velocity = min(self.velocity, self.max_velocity)
    #     elif 'backward' in self.vel_cmd:
    #         self.velocity -= self.acceleration / self.sim_fps
    #         self.velocity = max(self.velocity, -self.max_velocity)
    #     else:
    #         self.velocity = 0

    #     return self.velocity, self.steering
