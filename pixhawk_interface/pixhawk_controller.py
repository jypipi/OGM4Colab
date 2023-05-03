from pymavlink import mavutil
from typing import List
from data_class import *

class PixhawkInterface:
    def __init__(self, ip='192.168.2.1') -> None:
        self.master = mavutil.mavlink_connection(f'udpin:{ip}:14550')
        self.master.wait_heartbeat()
    

    def arm(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0)
    def disarm(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0)

    def wait_for_disarm(self):
        self.master.motors_disarmed_wait()
    def wait_for_arm(self):
        self.master.motors_armed_wait()

    def manual_control(self, sixd: List[float]):
        pass

    def request_param(self, param_id: str):
        param_bytes = bytes(param_id)
        self.master.mav.param_request_read_send(
            self.master.target_system, self.master.target_component,
            param_bytes,
            -1
        )
        message = self.master.recv_match(type='PARAM_VALUE', blocking=True).to_dict()['param_value']
        return message
    
    def get_data(self, data_type: str=None):
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            0,
            1, # 1Hz
            1, # start sending
            )

        msg = self.master.recv_match(type=data_type, blocking=True)

        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            0,
            1,
            1, # stop sending
            )

        if msg.get_type() != "BAD_DATA":
            msg = msg.to_dict()
            data_obj = locals()[data_type](msg)
            ROS_msg = data_obj.convert_2_ROS_msg()
            return ROS_msg