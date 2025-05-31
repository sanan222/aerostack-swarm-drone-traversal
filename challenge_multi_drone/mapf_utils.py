#!/usr/bin/env python3

import argparse
import sys
import time
import random
import threading
import math
from math import radians, cos, sin, atan2
from typing import List

import rclpy
from rclpy.executors import MultiThreadedExecutor
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA  # Add this import for LED control
import yaml  # Senaryo dosyasını okumak için
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D




############################################################
# Formation hesaplamaları için sadeleştirilmiş Choreographer #
############################################################

class Choreographer:
    @staticmethod
    def v_formation(length: float, angle_deg: float = 30.0,
                    orientation_deg: float = 0.0,
                    center: List[float] = [0.0, 0.0],
                    num_drones: int = 5) -> List[List[float]]:
        """
        Custom stepped formation:
        f1
            f2 
                 leader
             f3
        f4
        """
        theta = radians(orientation_deg)
        angle = radians(angle_deg)
        positions = []
        
        # Leader position (center front)
        leader = [center[0], center[1]]
        
        # Calculate direction vectors for left and right sides of the V
        left_dir_x = -cos(theta + angle)
        left_dir_y = -sin(theta + angle)
        
        right_dir_x = -cos(theta - angle) 
        right_dir_y = -sin(theta - angle)

        # Generate formation positions based on available drones
        if num_drones >= 5:
            # Leader at front point
            positions.append(leader.copy())
            
            # First row followers (on either side of the leader)
            left1 = [
                leader[0] + length * left_dir_x,
                leader[1] + length * left_dir_y
            ]
            
            right1 = [
                leader[0] + length * right_dir_x,
                leader[1] + length * right_dir_y
            ]
            
            # Second row followers (further back)
            left2 = [
                leader[0] + 2 * length * left_dir_x,
                leader[1] + 2 * length * left_dir_y
            ]
            
            right2 = [
                leader[0] + 2 * length * right_dir_x,
                leader[1] + 2 * length * right_dir_y
            ]
            
            # Arrange in desired order: left2 (f1), left1 (f2), leader, right1 (f3), right2 (f4)
            positions = [left2, left1, leader, right1, right2]
        elif num_drones == 4:
            positions = [left1, leader, right1, right2]
        elif num_drones == 3:
            positions = [left1, leader, right1]
        elif num_drones == 2:
            positions = [leader, right1]
        else:
            positions = [leader]
            
        return positions
        
    @staticmethod
    def pentagon_formation(radius: float, orientation_deg: float = 0.0,
                          center: List[float] = [0.0, 0.0],
                          num_drones: int = 5) -> List[List[float]]:
        """
        Generate a pentagon formation for drones
        
        Args:
            radius: Radius of the pentagon
            orientation_deg: Orientation of the pentagon in degrees
            center: Center of the pentagon [x, y]
            num_drones: Number of drones (should be 5 for a proper pentagon)
            
        Returns:
            List of positions for each drone in the formation
        """
        theta = radians(orientation_deg)
        positions = []
        
        # Calculate the angle between each point in the pentagon
        angle_step = 2 * math.pi / 5
        
        # Generate positions for each drone
        for i in range(min(num_drones, 5)):
            # Calculate angle for this point
            angle = i * angle_step - math.pi/2 + theta
            
            # Calculate position
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            
            positions.append([x, y])
        
        # If we have fewer than 5 drones, distribute them evenly around the pentagon
        if num_drones < 5:
            # Keep only the first num_drones positions
            positions = positions[:num_drones]
        
        return positions



#####################################
# Basitleştirilmiş Drone Sınıfı     #
#####################################

class SimpleDrone(DroneInterface):
    def __init__(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=False, use_sim_time=use_sim_time)
        self.path = []  # [ [x,y,z], ... ]
        self.path_index = 0
        self.current_pose = [0.0, 0.0, 0.0]
        self.is_moving = False
        # Create LED publisher with correct topic name
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        print(f"Initialized drone {self.drone_id}")

    def change_led_colour(self, colour):
        """Change the LED colors of the drone
        
        Args:
            colour (tuple): The LED RGB Colours (0-255)
        """
        msg = ColorRGBA()
        msg.r = colour[0]/255.0
        msg.g = colour[1]/255.0
        msg.b = colour[2]/255.0
        msg.a = 1.0  # Set alpha to 1.0
        self.led_pub.publish(msg)

    def change_leds_random_colour(self):
        """Change the LED colors to random colors"""
        self.change_led_colour([random.randint(0, 255) for _ in range(3)])

    def get_current_position(self):
        # Gerçek pozisyon alınmaya çalışılır, yoksa mevcut değeri döndürür.
        try:
            return self.position
        except Exception as e:
            return self.current_pose

    def set_path(self, path: List[List[float]]):
        self.path = path
        self.path_index = 0
        print(f"Path set for {self.drone_id} with {len(path)} waypoints.")

    def get_next_waypoint(self):
        if self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    def go_to_next(self):
        wp = self.get_next_waypoint()
        if wp is not None:
            print(f"{self.drone_id} going to waypoint: {wp}")
            speed = 0.5
            yaw_mode = YawMode.PATH_FACING
            try:
                # Execute go_to and wait for completion
                result = self.go_to(wp[0], wp[1], wp[2], speed, yaw_mode, None, "earth", True)
                self.is_moving = False
                self.current_pose = wp.copy()
                self.path_index += 1
                # Removed LED color change here to maintain consistent colors
                return True
            except Exception as e:
                print(f"Error in go_to for {self.drone_id}: {e}")
                self.is_moving = False
                return False
        else:
            print(f"{self.drone_id} no more waypoints.")
            return False

    def get_current_behavior_status(self):
        """Get the current behavior status of the drone"""
        try:
            # Call parent class method from DroneInterface
            return super().get_behavior_status()
        except Exception as e:
            print(f"Error getting behavior status: {e}")
            return None

    def goal_reached(self) -> bool:
        # Check if drone has reached its goal by querying behavior status
        try:
            status = super().get_behavior_status()  # Call parent method directly
            return status is None or status.status == BehaviorStatus.IDLE
        except Exception:
            return not self.is_moving

    def has_more_waypoints(self) -> bool:
        return self.path_index < len(self.path)

    def takeoff(self, height: float = 1.0):
        print(f"{self.drone_id} taking off to height {height}")
        try:
            result = super().takeoff(height, 0.5, False)
            self.is_moving = True
        except Exception as e:
            print(f"Takeoff error for {self.drone_id}: {e}")
        self.current_pose[2] = height

    def land(self):
        print(f"{self.drone_id} landing")
        try:
            result = super().land(0.5, False)
            self.is_moving = True
        except Exception as e:
            print(f"Land error for {self.drone_id}: {e}")
            self.is_moving = True


##################################
# 3D RRT* Path Planning Algorithm #
##################################

class RRTStar3D:
    """Advanced 3D RRT* path planning with rewiring and path smoothing"""

    class Node:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.parent = None
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dist=1.0, goal_sample_rate=10, max_iter=500,
                 rewire_radius=2.0, clearance=0.5):
        """
        Args:
            start: [x, y, z]
            goal: [x, y, z]
            obstacle_list: List of obstacles as [x, y, z, width, depth, height]
            rand_area: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        self.start = self.Node(*start)
        self.goal = self.Node(*goal)
        self.obstacle_list = obstacle_list
        self.min_x, self.max_x = rand_area[0], rand_area[1]
        self.min_y, self.max_y = rand_area[2], rand_area[3]
        self.min_z, self.max_z = rand_area[4], rand_area[5]
        self.expand_dist = expand_dist
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.rewire_radius = rewire_radius
        self.clearance = clearance
        self.node_list = [self.start]

    def plan(self):
        for i in range(self.max_iter):
            # 1. Generate random node
            if random.randint(0, 100) < self.goal_sample_rate:
                rnd = self.Node(self.goal.x, self.goal.y, self.goal.z)
            else:
                rnd = self.Node(
                    random.uniform(self.min_x, self.max_x),
                    random.uniform(self.min_y, self.max_y),
                    random.uniform(self.min_z, self.max_z)
                )
            
            # 2. Find nearest node
            nearest_ind = self._get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # 3. Steer toward random node
            new_node = self._steer(nearest_node, rnd, self.expand_dist)
            
            # 4. Check collision
            if self._check_collision(new_node, nearest_node):
                # 5. Find nearby nodes for rewiring
                nearby_nodes = self._find_near_nodes(new_node)
                
                # 6. Choose best parent
                self._choose_parent(new_node, nearby_nodes)
                
                # 7. Add node to tree
                self.node_list.append(new_node)
                
                # 8. Rewire tree
                self._rewire(new_node, nearby_nodes)
                
                # 9. Check if we can connect to goal
                if self._is_near_goal(new_node):
                    if self._check_collision(new_node, self.goal):
                        final_node = self.Node(self.goal.x, self.goal.y, self.goal.z)
                        final_node.parent = new_node
                        final_node.cost = new_node.cost + self._calc_distance(new_node, final_node)
                        self.node_list.append(final_node)
                        break
        
        # Extract and smooth path
        path = self._extract_path()
        smoothed_path = self._smooth_path(path)
        return smoothed_path

    def _get_nearest_node_index(self, node):
        """Find index of nearest node in the tree"""
        dists = [(n.x - node.x) ** 2 + 
                 (n.y - node.y) ** 2 + 
                 (n.z - node.z) ** 2 for n in self.node_list]
        return dists.index(min(dists))

    def _steer(self, from_node, to_node, extend_length=float("inf")):
        """Create new node by steering from 'from_node' toward 'to_node'"""
        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d = self._calc_distance(from_node, to_node)
        
        if d > extend_length:
            # Limit distance to extend_length
            ratio = extend_length / d
            new_node.x = from_node.x + ratio * (to_node.x - from_node.x)
            new_node.y = from_node.y + ratio * (to_node.y - from_node.y)
            new_node.z = from_node.z + ratio * (to_node.z - from_node.z)
        else:
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z
        
        new_node.parent = from_node
        new_node.cost = from_node.cost + self._calc_distance(from_node, new_node)
        
        return new_node

    def _calc_distance(self, from_node, to_node):
        """Calculate Euclidean distance between nodes"""
        return math.sqrt((from_node.x - to_node.x) ** 2 + 
                         (from_node.y - to_node.y) ** 2 + 
                         (from_node.z - to_node.z) ** 2)

    def _check_collision(self, from_node, to_node, samples=10):
        """Check if the path between nodes collides with obstacles"""
        for i in range(samples + 1):
            t = i / samples
            x = from_node.x + (to_node.x - from_node.x) * t
            y = from_node.y + (to_node.y - from_node.y) * t
            z = from_node.z + (to_node.z - from_node.z) * t
            
            for ox, oy, oz, width, depth, height in self.obstacle_list:
                # Check collision with box obstacle
                half_width = width / 2
                half_depth = depth / 2
                half_height = height / 2
                
                if (abs(x - ox) <= half_width + self.clearance and
                    abs(y - oy) <= half_depth + self.clearance and
                    abs(z - oz) <= half_height + self.clearance):
                    return False
                    
        return True

    def _find_near_nodes(self, node):
        """Find nodes within rewiring radius"""
        indices = []
        for i, n in enumerate(self.node_list):
            if self._calc_distance(node, n) <= self.rewire_radius:
                indices.append(i)
        return indices

    def _choose_parent(self, node, nearby_indices):
        """Choose best parent for the new node"""
        if not nearby_indices:
            return
        
        costs = []
        for i in nearby_indices:
            near_node = self.node_list[i]
            potential_cost = near_node.cost + self._calc_distance(near_node, node)
            costs.append(potential_cost)
            
        min_cost_index = costs.index(min(costs))
        min_cost = costs[min_cost_index]
        
        # Only update if better and collision-free
        if min_cost < node.cost:
            near_node = self.node_list[nearby_indices[min_cost_index]]
            if self._check_collision(near_node, node):
                node.parent = near_node
                node.cost = min_cost

    def _rewire(self, new_node, nearby_indices):
        """Rewire tree by checking if paths through new_node are better"""
        for i in nearby_indices:
            near_node = self.node_list[i]
            edge_cost = self._calc_distance(new_node, near_node)
            new_cost = new_node.cost + edge_cost
            
            if new_cost < near_node.cost:
                if self._check_collision(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def _is_near_goal(self, node):
        """Check if node is close to goal"""
        return self._calc_distance(node, self.goal) <= self.expand_dist

    def _extract_path(self):
        """Extract path from goal node to start node"""
        path = []
        
        # If goal node is in the tree
        if self.goal.parent:
            node = self.goal
        else:
            # Find node closest to goal
            min_dist = float('inf')
            closest_node = None
            for node in self.node_list:
                dist = self._calc_distance(node, self.goal)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node
            node = closest_node
        
        # Trace back to start
        while node:
            path.append([node.x, node.y, node.z])
            node = node.parent
            
        # Reverse path (start to goal)
        return path[::-1]

    def _smooth_path(self, path, max_trials=100):
        """Apply path shortcutting/smoothing"""
        if len(path) < 3:
            return path
            
        for _ in range(max_trials):
            # Pick two random indices
            i = random.randint(0, len(path) - 3)
            j = random.randint(i + 2, len(path) - 1)
            
            # Check if shortcut is collision-free
            start_node = self.Node(*path[i])
            end_node = self.Node(*path[j])
            
            if self._check_collision(start_node, end_node, samples=20):
                # If collision-free, create shortcut
                path = path[:i+1] + path[j:]
                
        return path

def plot_cube(ax, center, size, color='red', alpha=0.3):
    """Plot a 3D cube/box on the given axis"""
    cx, cy, cz = center
    dx, dy, dz = size
    
    # Define the 8 corners of the cube
    x = [cx - dx/2, cx + dx/2]
    y = [cy - dy/2, cy + dy/2]
    z = [cz - dz/2, cz + dz/2]
    
    corners = [(xi, yi, zi) for xi in x for yi in y for zi in z]
    
    # Define the 6 faces using indices to corners
    faces_idx = [
        [0, 1, 3, 2],  # Bottom face
        [4, 5, 7, 6],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [0, 2, 6, 4],  # Left face
        [1, 3, 7, 5],  # Right face
    ]
    
    faces = [[corners[i] for i in face] for face in faces_idx]
    
    # Plot faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))
