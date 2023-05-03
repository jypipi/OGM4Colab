## Create a new file pixhawk_controller.py PixhawkController class with abstractions for the mavlink calls
## Subscribes to /cmd_vel
## Arm / disarm service
## publishes whatever sensors are connected to the pixhawk
import rclpy
from interfaces.srv import ArmPixhawk, DisarmPixhawk
import pixhawk_controller

class PixhawkInterfaceNode(rclpy.Node):
    def __init__(self) -> None:
        super().__init__('PixhawkInterfaceNode')
        self.pixhawk = pixhawk_controller.PixhawkInterface()
        self.armsrv = self.create_service(ArmPixhawk, 'arm_pixhawk', self.arm_pixhawk_cb)
        self.disarmsrv = self.create_service(DisarmPixhawk, 'disarm_pixhawk', self.disarm_pixhawk_cb)

    def arm_pixhawk_cb(self, request, response):
        response.success = False
        self.pixhawk.arm()
        self.pixhawk.wait_for_arm()
        response.success = True
        return response
    
    def disarm_pixhawk_cb(self, request, response):
        response.success = False
        self.pixhawk.disarm()
        self.pixhawk.wait_for_disarm()
        response.success = True
        return response



def main(args=None):
    rclpy.init(args=args)
    pixhawk_interface_node = PixhawkInterfaceNode()
    rclpy.spin(pixhawk_interface_node)
    rclpy.shutdown()



if __name__ == '__main__':
    main()
