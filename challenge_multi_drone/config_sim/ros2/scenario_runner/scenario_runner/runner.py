import os
from typing import Dict, List
import numpy as np
from functools import partial
import yaml
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray

import copy


class ScenarioObstacleViz(Node):

    def __init__(self):
        super().__init__("scenario_obstacle_viz")

        self.declare_parameter("scenario_file", rclpy.Parameter.Type.STRING)
        self.declare_parameter("frequency", 20) # 20hz

        self.scenario = self.read_yaml(self.get_parameter("scenario_file").value)

        if "stage4" not in self.scenario:
            self.get_logger().info("Stage 4 not present in the current scenario, exiting...")
            return

        self.dynamic_obstacles_pub_locations = self.create_publisher(PoseStamped, "/dynamic_obstacles/locations", 10)

        self.timer = self.create_timer(1.0/self.get_parameter("frequency").value, self.dynamic_obs_control_cb)
        self.current_time = self.get_clock().now()

        self.initialise_obstacles()

        self.get_logger().info(f"Scenario Obstacle Runner Intialised for {self.get_parameter('scenario_file').value}")

    def initialise_obstacles(self):

        self.size = self.scenario["stage_size"]
        self.stage = self.scenario["stage4"]

        self.num_obs = self.stage["num_obstacles"]

        self.center = self.stage["stage_center"]
        self.boundaries = {
                "min_x": self.center[0] - self.size[0]/2.0,
                "min_y": self.center[1] - self.size[1]/2.0,
                "max_x": self.center[0] + self.size[0]/2.0,
                "max_y": self.center[1] + self.size[1]/2.0
            }

        
        self.init_angles = np.random.random(self.num_obs) * 2 * np.pi
        self.velocities = self.stage["obstacle_velocity"] * np.array([np.array([np.cos(a), np.sin(a)]) for a in self.init_angles])
        self.locations = []
        for _ in range(self.num_obs):
            # Get random start location relative to stage center and stage size
            loc = [l + (random.random() * s) - (s/2.0) for s, l in zip(self.size, self.center)]    
            self.locations.append(np.array(loc))
    
    def dynamic_obs_control_cb(self):
        self.this_time = self.get_clock().now()
        dt = self.this_time - self.current_time

        for i in range(self.num_obs):

            msg = PoseStamped()
            msg.header.stamp = self.this_time.to_msg()
            msg.header.frame_id = f"object_{i}"

            new_pos = self.locations[i] + self.velocities[i] * dt.nanoseconds * 1e-9

            if new_pos[0] < self.boundaries["min_x"] or new_pos[0] > self.boundaries["max_x"]:
                self.velocities[i][0] = -self.velocities[i][0]
                new_pos[0] = np.clip(new_pos[0], self.boundaries["min_x"], self.boundaries["max_x"])

            if new_pos[1] < self.boundaries["min_y"] or new_pos[1] > self.boundaries["max_y"]:
                self.velocities[i][1] = -self.velocities[i][1]
                new_pos[1] = np.clip(new_pos[1], self.boundaries["min_y"], self.boundaries["max_y"])

            self.locations[i] = new_pos

            msg.pose.position.x = new_pos[0]
            msg.pose.position.y = new_pos[1]
            msg.pose.position.z = 0.0
            
            self.dynamic_obstacles_pub_locations.publish(msg)

        self.current_time = self.this_time

    # Read the YAML scenario file
    def read_yaml(self, file_path):
        with open(file_path, 'r') as file:
            scenario = yaml.safe_load(file)
        return scenario

def main(args=None):
    rclpy.init(args=args)

    drilling_monitor = ScenarioObstacleViz()

    rclpy.spin(drilling_monitor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    drilling_monitor.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
