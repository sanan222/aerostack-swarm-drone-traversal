import os
from typing import Dict, List
import numpy as np
from functools import partial
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

import copy


class ScenarioObstacleViz(Node):

    def __init__(self):
        super().__init__("scenario_obstacle_viz")

        self.declare_parameter("scenario_file", rclpy.Parameter.Type.STRING)
        self.declare_parameter("stage_4_obstacle_topic", "/dynamic_obstacles/locations")

        self.scenario = self.read_yaml(self.get_parameter("scenario_file").value)

        self.dynamic_obstacle_sub = self.create_subscription(PoseStamped,
                                                             self.get_parameter("stage_4_obstacle_topic").value,
                                                             self.dynamic_obs_cb, 10)
        latching_qos = QoSProfile(depth=1)
        latching_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.viz_markers_pub = self.create_publisher(MarkerArray, "viz/obstacles", latching_qos)
        self.viz_markers_dynamic_pub = self.create_publisher(Marker, "viz/dynamic_obstacles", 10)

        if "stage4" in self.scenario:
            self.tracking_dict = {}
            self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info(f"Scenario Obstacle Viz Intialised for {self.get_parameter('scenario_file').value}")

        self.latch_publish()

    def timer_cb(self):

        stage = self.scenario.get("stage4", {})

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "earth"

        marker.ns = "dynamic_object"
       
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        marker.scale.x = stage["obstacle_diameter"] #/ 10.0
        marker.scale.z = stage["obstacle_height"]    #/ 10.0
        marker.scale.y = stage["obstacle_diameter"] #/ 10.0

        colour = [1.0, 0.5, 0.5, 1.0]
        marker.color.a = colour[0]
        marker.color.r = colour[1]
        marker.color.g = colour[2]
        marker.color.b = colour[3]

        for id, point in self.tracking_dict.items():
            marker.id = id
            point.z = stage["obstacle_height"] / 2.0
            marker.points.append(point)

        self.viz_markers_dynamic_pub.publish(marker)
    
    def dynamic_obs_cb(self, msg):
        msg_id = int(msg.header.frame_id.split("_")[1])
        self.tracking_dict[msg_id] = msg.pose.position
    
    # Read the YAML scenario file
    def read_yaml(self, file_path):
        with open(file_path, 'r') as file:
            scenario = yaml.safe_load(file)
        return scenario
    

    def generate_as2_windows_config(self):
        markers = []        
        window_list = self.scenario["stage2"]["windows"]
        stage = self.scenario["stage2"]["stage_center"]

        # Add obstacles as objects
        for key, w in window_list.items():
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "earth"

            marker.ns = "window"
            marker.id = key

            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.scale.x = w["gap_width"] #/ 10.0
            marker.scale.z = w["height"]    #/ 10.0
            marker.scale.y = w["thickness"] #/ 10.0

            marker.pose.position.x = stage[0] + w["center"][0]     
            marker.pose.position.y = stage[1] + w["center"][1] 
            marker.pose.position.z = w["distance_floor"] + w["height"]/2.0 

            colour = [1.0, 1.0, 0.5, 0.3]
            marker.color.a = colour[0]
            marker.color.r = colour[1]
            marker.color.g = colour[2]
            marker.color.b = colour[3]

            markers.append(marker)
        return markers  
    
    def generate_as2_forest_config(self):
        markers = []

        stage = self.scenario.get("stage3", {})
        stage_center = stage["stage_center"]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "earth"

        marker.ns = "tree_columns"
        marker.id = 0

        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        marker.scale.x = stage["obstacle_diameter"] #/ 10.0
        marker.scale.z = stage["obstacle_height"]    #/ 10.0
        marker.scale.y = stage["obstacle_diameter"] #/ 10.0

        colour = [1.0, 1.0, 1.0, 1.0]
        marker.color.a = colour[0]
        marker.color.r = colour[1]
        marker.color.g = colour[2]
        marker.color.b = colour[3]

        # Add obstacles as objects
        for key, coord in enumerate(stage["obstacles"]):
            p = Point()
            p.x = stage_center[0] + coord[0]     
            p.y = stage_center[1] + coord[1] 
            p.z = stage["obstacle_height"] / 2.0
            marker.points.append(p)

        markers.append(marker)
        return markers  
    
    def generate_as2_floor_tiles_config(self):
        markers = []
        size = self.scenario["stage_size"]

        mapping = {
            "stage1": (1.0, 0.0, 0.4, 0.5),
            "stage2": (0.1, 1.0, 0.4, 0.5),
            "stage3": (0.1, 0.0, 0.5, 0.5),
            "stage4": (1.0, 0.0, 1.0, 0.5),
        }

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "earth"

        marker.ns = "floor"
        marker.id = 0

        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        marker.scale.x = size[0]
        marker.scale.z = 0.1
        marker.scale.y = size[1]
        
        for stage, colour in mapping.items():
            if stage not in self.scenario:
                continue
            
            c = ColorRGBA()
            c.a = colour[3]
            c.r = colour[0]
            c.g = colour[1]
            c.b = colour[2]
            marker.colors.append(c)

            # Add obstacles as objects
            loc = self.scenario[stage]["stage_center"]
            p = Point()
            p.x = loc[0]
            p.y = loc[1]
            p.z = 0.0
            marker.points.append(p)

        markers.append(marker)
        return markers  

    def latch_publish(self):

        msg = MarkerArray()

        msg.markers.extend(self.generate_as2_floor_tiles_config())

        if "stage2" in self.scenario:
            msg.markers.extend(self.generate_as2_windows_config())

        if "stage3" in self.scenario:
            msg.markers.extend(self.generate_as2_forest_config())
        
        self.viz_markers_pub.publish(msg)
        self.get_logger().info(f"Published latch for obstacle viz")



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
