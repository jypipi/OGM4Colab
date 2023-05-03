class ATTITUDE:
    def __init__(self, msg):
        # self.time_stamp = msg["time_boot_ms"]

        # self.roll = msg["roll"]
        # self.pitch = msg["pitch"]
        # self.yaw = msg["yaw"]

        # self.rollspeed = msg["rollspeed"]
        # self.pitchspeed = msg["pitchspeed"]
        # self.yawspeed = msg["yawspeed"]

        self.msg = "Hello from ATT obj"

    def convert_2_ROS_msg(self):
        print(self.msg)

class SCALED_IMU:
    def __init__(self, msg):
        self.time_stamp = msg["time_boot_ms"]

        self.x_acceleration = msg["xacc"]
        self.y_acceleration = msg["yacc"]
        self.z_acceleration = msg["zacc"]

        self.x_angular_speed = msg["xgyro"]
        self.y_angular_speed = msg["ygyro"]
        self.z_angular_speed = msg["zgyro"]

    def convert_2_ROS_msg(self):
        pass