import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from rokubimini_msgs.srv import ResetWrench

class ForceSensorResetNode(Node):
    def __init__(self):
        super().__init__('force_sensor_reset_node')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'data_control',
            self.listener_callback,
            10)
        self.client = self.create_client(ResetWrench, '/bus0/ft_sensor0/reset_wrench')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.request = ResetWrench.Request()
        self.request.desired_wrench.force.x = 0.0
        self.request.desired_wrench.force.y = 0.0
        self.request.desired_wrench.force.z = 0.0
        self.request.desired_wrench.torque.x = 0.0
        self.request.desired_wrench.torque.y = 0.0
        self.request.desired_wrench.torque.z = 0.0
        self.flag = True


    def listener_callback(self, msg):
        if len(msg.data) > 4 and (int(msg.data[5]) % 19 == 0 and self.flag) and int(msg.data[5]) != 0:
            self.reset_force_sensor()
            self.flag = False
            self.get_logger().info('Force sensor reset requested with data: {}'.format(msg.data))
        if len(msg.data) > 4 and (int(msg.data[5]) % 19 != 0):
            self.flag = True

    def reset_force_sensor(self):
        future = self.client.call_async(self.request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().info('Reset failed, trying again...')
                self.reset_force_sensor()
            else:
                self.get_logger().info('Force sensor reset successfully.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            
            self.reset_force_sensor()

def main(args=None):
    rclpy.init(args=args)
    node = ForceSensorResetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()