from time import time
from math import pi

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.reference = None
        self.I_term = 0.0
        
        self.last_error = 0.0
        # self.last_time = time()
        
    def adjust(self, reference, feedback, delta_time):
        # curr_time = time()
        # delta_time = curr_time - self.last_time

        curr_error = reference - feedback

        P_term = self.Kp * curr_error
        self.I_term = curr_error * delta_time
        
        D_term = 0.0
        if delta_time > 0:
            D_term = self.Kd * (curr_error-self.last_error) / delta_time

        self.last_error = curr_error
        # self.last_time = curr_time

        adjustment = P_term + self.Ki * self.I_term + D_term
        # if abs(adjustment) > abs(2*pi - adjustment):
        #     adjustment = -(2*pi - adjustment)

        return adjustment
