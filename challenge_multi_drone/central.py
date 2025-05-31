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


#####################################
# Swarm Conductor - Koordinasyon      #
#####################################

class SwarmConductor:
    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False, scenario_file: str = None):
        self.num_drones = len(drones_ns)
        self.drones = {}
        for idx, ns in enumerate(drones_ns):
            self.drones[idx] = SimpleDrone(ns, verbose, use_sim_time)
        self.leader_index = 0  # İlk drone lider olarak atanır
        self.formation_length_first_last = 0.75
        self.formation_length = 0.35  # Reduced spacing between drones (was 0.8)
        self.obstacles = []  # [x, y, z, radius]
        self.boundaries = [-10, 10, -10, 10, 0, 5]
        self.scenario_data = None
        if scenario_file:
            try:
                with open(scenario_file, "r") as f:
                    scenario_data = yaml.safe_load(f)
                self.scenario_data = scenario_data
                if "obstacles" in scenario_data:
                    self.obstacles = scenario_data["obstacles"]
                print(f"Scenario file loaded: {scenario_file}")
            except Exception as e:
                print(f"Error loading scenario file: {e}")

    def generate_formation_positions(self, leader_position, orientation, scale=1.0):
        """
        Generate formation positions for all drones based on leader position
        Now includes a scale parameter to adjust formation size
        """
        center = leader_position[:2]
        
        # Generate formation positions based on current formation type
        if getattr(self, "using_wide_formation", True):
            # Use wider formation with larger angle and distance
            formation_xy = Choreographer.v_formation(self.formation_length_first_last, 25.0, orientation, center, self.num_drones)
        else:
            # Use tighter formation with smaller angle and distance
            formation_xy = Choreographer.v_formation(self.formation_length, 20.0, orientation, center, self.num_drones)
        
        # Add z-coordinate from leader's position
        formation_positions = [pos + [leader_position[2]] for pos in formation_xy]
        
        # Reorder so leader is first (leader is at index 2 in the formation)
        leader_position = formation_positions[2]
        follower_positions = [p for i, p in enumerate(formation_positions) if i != 2]
        
        # Reorder so leader is first, then followers
        reordered_positions = [leader_position] + follower_positions
        
        # Apply scale to all relative positions
        if scale != 1.0:
            for i in range(1, len(reordered_positions)):
                # Skip leader (index 0)
                # Apply scale to relative position from leader
                rel_x = reordered_positions[i][0] - leader_position[0]
                rel_y = reordered_positions[i][1] - leader_position[1]
                
                # Scale the relative position
                reordered_positions[i][0] = leader_position[0] + rel_x * scale
                reordered_positions[i][1] = leader_position[1] + rel_y * scale
        
        return reordered_positions

    def generate_paths_for_all_drones(self, leader_path: List[List[float]]):
        """Liderin yol noktaları üzerinden, takipçilerin yollarını oluştur."""
        leader = self.drones[self.leader_index]
        leader.set_path(leader_path)

        # Takipçi yolları için boş bir liste oluştur
        follower_paths = {idx: [] for idx in self.drones if idx != self.leader_index}

        # Use the fixed formation orientation established during initial formation
        orientation_deg = self.formation_orientation
        
        # Her lider waypoint'i için formation pozisyonlarını hesapla
        for i in range(len(leader_path)):
            # Use fixed orientation for the formation regardless of movement direction
            formation_positions = self.generate_formation_positions(leader_path[i], orientation_deg)
            # Lider pozisyonu zaten leader_path'te, takipçiler için formation_positions'dan diğerleri kullanılır
            follower_index = 0
            for drone_idx in follower_paths.keys():
                if (follower_index + 1) < len(formation_positions):
                    pos = formation_positions[follower_index + 1]
                else:
                    pos = formation_positions[-1]
                # Çarpışma kontrolü: engel yakınındaysa küçük offset ekle
                safe = True
                for obs in self.obstacles:
                    ox, oy, oz, radius = obs
                    d = math.sqrt((pos[0]-ox)**2 + (pos[1]-oy)**2 + (pos[2]-oz)**2)
                    if d < radius:
                        safe = False
                        break
                if not safe:
                    offset = 0.5
                    pos = [
                        pos[0] + offset * cos(radians(orientation_deg)),
                        pos[1] + offset * sin(radians(orientation_deg)),
                        pos[2]
                    ]
                follower_paths[drone_idx].append(pos)
                follower_index += 1

        for idx, path in follower_paths.items():
            self.drones[idx].set_path(path)

    def takeoff_all(self):
        # First arm and set to offboard mode for all drones
        for drone in self.drones.values():
            drone.arm()
            drone.offboard()
        
        # Set LED color to green for all drones
        for drone in self.drones.values():
            drone.change_led_colour((0, 255, 0))  # Green for takeoff
        
        # Send takeoff commands to all drones without waiting (non-blocking)
        print("Sending takeoff commands to all drones simultaneously...")
        for drone in self.drones.values():
            # Use non-blocking takeoff (False parameter)
            drone.takeoff(1.5, 0.5, False)
        
        # Wait for all drones to complete takeoff
        print("Waiting for all drones to reach takeoff altitude...")
        all_reached_altitude = False
        max_wait_time = 15  # Maximum wait time in seconds
        start_time = time.time()
        
        while not all_reached_altitude and time.time() - start_time < max_wait_time:
            all_reached_altitude = True
            for drone in self.drones.values():
                try:
                    # Check if drone has reached target altitude
                    current_altitude = drone.position[2]
                    if current_altitude < 1.4:  # Slightly below target to account for variations
                        all_reached_altitude = False
                        break
                except Exception:
                    all_reached_altitude = False
                    break
            
            if not all_reached_altitude:
                time.sleep(0.2)  # Short sleep before checking again
        
        print("All drones have taken off.")
        
        # Give an additional delay for stability
        time.sleep(2)

    def form_initial_formation(self):
        """Position drones in the initial V formation before starting the mission, facing the window"""
        print("Forming initial V formation...")
        
        # Get current leader position
        leader = self.drones[self.leader_index]
        leader_pos = leader.get_current_position()
        
        # Determine direction based on which scenario we're running
        formation_orientation = 270.0  # Default orientation (facing negative Y / south)
        
        # Check if we're running stage4 (dynamic obstacle avoidance)
        if self.scenario_data and "stage4" in self.scenario_data:
            # For stage4, face north (positive Y)
            formation_orientation = 90.0
            print(f"Stage 4 detected: Setting formation orientation to North ({formation_orientation} degrees)")
        elif self.scenario_data and ("stage1" in self.scenario_data or 
                                    "stage2" in self.scenario_data or 
                                    "stage3" in self.scenario_data):
            # For stages 1-3, face south (negative Y)
            formation_orientation = 270.0
            print(f"Stage 1-3 detected: Setting formation orientation to South ({formation_orientation} degrees)")
        
        # Store the formation orientation for use in later mission phases
        self.formation_orientation = formation_orientation
        
        # Generate formation positions based on leader's position and orientation toward window
        formation_positions = self.generate_formation_positions(leader_pos, formation_orientation)
        
        # First move drones vertically to different altitudes to avoid collisions
        follower_index = 0
        altitude_offsets = [0.3, 0.6, 0.9, 1.2]  # Different altitude offsets
        active_drones = []
        
        for idx, drone in self.drones.items():
            if idx != self.leader_index:
                # Move each follower to a different altitude first
                altitude = leader_pos[2] + altitude_offsets[follower_index % len(altitude_offsets)]
                intermediate_pos = [drone.get_current_position()[0], drone.get_current_position()[1], altitude]
                print(f"Moving {drone.drone_id} to intermediate altitude: {altitude}m")
                drone.go_to(intermediate_pos[0], intermediate_pos[1], intermediate_pos[2], 0.5, YawMode.FIXED_YAW, formation_orientation, "earth", False)
                active_drones.append(drone)
                follower_index += 1
        
        # Wait for altitude changes to complete
        time.sleep(3)
        
        # Now move all drones to their horizontal positions simultaneously
        follower_index = 0
        active_drones = []
        
        # First send all drones to their target positions at their current altitude
        for idx, drone in self.drones.items():
            if idx != self.leader_index:
                pos = formation_positions[follower_index + 1]  # +1 because leader position is at index 0
                current_pos = drone.get_current_position()
                # Move horizontally while maintaining current altitude
                intermediate_pos = [pos[0], pos[1], current_pos[2]]
                print(f"Moving {drone.drone_id} to formation position (horizontally): {intermediate_pos}")
                drone.go_to(intermediate_pos[0], intermediate_pos[1], intermediate_pos[2], 0.5, YawMode.FIXED_YAW, formation_orientation, "earth", False)
                active_drones.append(drone)
                follower_index += 1
        
        # Wait for horizontal movements to complete
        time.sleep(5)
        
        # Finally, adjust all drones to the target formation altitude
        follower_index = 0
        active_drones = []
        
        for idx, drone in self.drones.items():
            if idx != self.leader_index:
                pos = formation_positions[follower_index + 1]  # +1 because leader position is at index 0
                print(f"Moving {drone.drone_id} to final formation altitude: {pos[2]}m")
                drone.go_to(pos[0], pos[1], pos[2], 0.3, YawMode.FIXED_YAW, formation_orientation, "earth", False)
                active_drones.append(drone)
                follower_index += 1
        
        # Also set leader yaw to face the window
        leader.go_to(leader_pos[0], leader_pos[1], leader_pos[2], 0.3, YawMode.FIXED_YAW, formation_orientation, "earth", False)
        active_drones.append(leader)
        
        # Wait for all drones to complete their movements
        time.sleep(5)
        
        # Set consistent LED colors for leader and followers for visualization
        leader.change_led_colour((255, 0, 0))  # Red for leader
        for idx, drone in self.drones.items():
            if idx != self.leader_index:
                drone.change_led_colour((0, 0, 255))  # Blue for followers
        
        print("Initial V formation complete")
        time.sleep(2)  # Brief pause to stabilize formation

    def execute_scenario1_mission(self):
        print("Executing Scenario 1 mission...")


    def execute_scenario2_mission(self):
        """
        Scenario 2: Window Traversal
         - Uses waypoints calculated from the scenario file
         - Leader follows these waypoints while followers maintain formation
         - Transitions between wide and tight formations at key points
        """
        leader_waypoints = []

        if self.scenario_data and "stage2" in self.scenario_data:
            stage2       = self.scenario_data["stage2"]
            y0, x0       = stage2["stage_center"]  # (y, x)
            windows      = stage2["windows"]
            window_ids   = sorted(windows.keys(), key=lambda k: int(k))
            offset       = 3.0

            # Track previous waypoint for midpoint calculation
            prev_waypoint = None

            for i, wid in enumerate(window_ids):
                w     = windows[wid]
                wy, wx = w["center"]                  # (y, x)
                gz     = w["distance_floor"] + w["height"] / 2.0

                gy = y0 + wy
                gx = x0 + wx

                # Add appropriate waypoints based on window position
                if i == 0:
                    # First window: add starting point 3m before window
                    leader_waypoints.append([gy, gx + offset, gz])  # Start point
                    leader_waypoints.append([gy, gx, gz])          # Window 1 waypoint
                    prev_waypoint = [gy, gx, gz]
                else:
                    current_waypoint = [gy, gx, gz]
                    
                    # Calculate and add midpoint between consecutive window waypoints
                    if prev_waypoint:
                        # Keep x and z from previous waypoint, only calculate mid y
                        mid_y = prev_waypoint[0]
                        mid_x = prev_waypoint[1] + (gx - prev_waypoint[1]) / 2
                        mid_z = prev_waypoint[2]
                        leader_waypoints.append([mid_y, mid_x, mid_z])  # Add midpoint waypoint
                    
                    # Now add the window waypoint
                    leader_waypoints.append(current_waypoint)
                    prev_waypoint = current_waypoint  # Update previous waypoint

                # son pencerenin 3m sonrası -> end
                if i == len(window_ids) - 1:
                    leader_waypoints.append([gy, gx - offset, gz])
        
        # If we couldn't get waypoints from the file, use defaults
        if not leader_waypoints:
            print("Warning: Could not calculate waypoints from scenario file. Using defaults.")
            leader_waypoints = [
                [-0.5, -1.5, 2.0],   # Start
                [-0.5, -4.5, 2.0],   # Window 1
                [-0.5, -6.5, 2.0],   # Midpoint between Window 1 and 2 (only y changes)
                [ 1.0, -8.5, 3.75],  # Window 2
                [ 1.0, -11.5, 3.75]  # End
            ]
        
        print("Leader waypoints for Scenario 2 (from YAML):", leader_waypoints)

        # Use waypoints directly without RRT* path planning
        leader_path = leader_waypoints
        print(f"Using {len(leader_path)} waypoints for leader.")
        
        # Initialize with wider formation for initial positioning
        self.using_wide_formation = True
        self.generate_paths_for_all_drones(leader_path)

        leader = self.drones[self.leader_index]
        followers = [d for idx, d in self.drones.items() if idx != self.leader_index]
        all_drones = [leader] + followers
        
        # Use the fixed formation orientation for all movements
        fixed_orientation = self.formation_orientation
        
        # Process waypoints one by one
        current_waypoint_index = 0
        
        # Set consistent colors for leader and followers
        leader.change_led_colour((255, 0, 0))  # Red for leader
        for follower in followers:
            follower.change_led_colour((0, 0, 255))  # Blue for followers
        
        while current_waypoint_index < len(leader_path):
            print(f"\n--- Moving to waypoint {current_waypoint_index+1}/{len(leader_path)} ---")
            
            # Move all drones to their respective waypoints at the current index
            active_drones = []
            
            # Start leader movement with fixed yaw orientation
            print(f"Leader {leader.drone_id}: current pos: {leader.get_current_position()}, target waypoint: {leader_path[current_waypoint_index]}")
            leader.is_moving = True
            leader.go_to(
                leader_path[current_waypoint_index][0], 
                leader_path[current_waypoint_index][1], 
                leader_path[current_waypoint_index][2], 
                0.3, YawMode.FIXED_YAW, fixed_orientation,
                "earth", False)
            active_drones.append(leader)
            
            # Start followers' movements with fixed yaw orientation
            for i, follower in enumerate(followers):
                if current_waypoint_index < len(follower.path):
                    target_wp = follower.path[current_waypoint_index]
                    print(f"Follower {follower.drone_id}: current pos: {follower.get_current_position()}, target waypoint: {target_wp}")
                    follower.is_moving = True
                    follower.go_to(
                        target_wp[0], target_wp[1], target_wp[2], 
                        0.3, YawMode.FIXED_YAW, fixed_orientation,
                        "earth", False)
                    active_drones.append(follower)
                else:
                    print(f"Warning: Follower {follower.drone_id} has no waypoint at index {current_waypoint_index}")
            
            # Wait for all drones to complete their movements
            print(f"Waiting for all {len(active_drones)} drones to reach waypoint {current_waypoint_index+1}...")
            start_time = time.time()
            max_wait = 60  # Increased maximum wait time in seconds (was 30)
            
            while time.time() - start_time < max_wait and active_drones:
                # Check each active drone
                for drone in active_drones[:]:  # Use a copy for safe iteration while removing
                    reached_waypoint = False
                    
                    # First try behavior status check
                    try:
                        status = super(SimpleDrone, drone).get_behavior_status()
                        if status is None or status.status == BehaviorStatus.IDLE:
                            reached_waypoint = True
                    except Exception:
                        pass
                        
                    # If behavior status check doesn't confirm arrival, check position
                    if not reached_waypoint:
                        try:
                            current_pos = drone.position
                            if current_waypoint_index < len(drone.path):
                                target_pos = drone.path[current_waypoint_index]
                                dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                              (current_pos[1]-target_pos[1])**2 + 
                                              (current_pos[2]-target_pos[2])**2)
                                # Use a slightly larger threshold to account for hovering variations
                                if dist < 0.3:  # Within 30cm is considered reached
                                    reached_waypoint = True
                        except Exception as e:
                            print(f"Error checking position for {drone.drone_id}: {e}")
                    
                    # If drone has reached waypoint by either method, mark as complete
                    if reached_waypoint:
                        print(f"{drone.drone_id} reached waypoint {current_waypoint_index+1}.")
                        drone.is_moving = False
                        # Update current position to exactly match the waypoint
                        if current_waypoint_index < len(drone.path):
                            drone.current_pose = drone.path[current_waypoint_index].copy()
                        # Maintain consistent LED colors instead of random
                        if drone == leader:
                            drone.change_led_colour((255, 0, 0))  # Red for leader
                        else:
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                        active_drones.remove(drone)
                
                # Short sleep between checks
                if active_drones:
                    time.sleep(0.5)
            
            # If any drones didn't reach their waypoints, handle the timeout
            if active_drones:
                print(f"Warning: {len(active_drones)} drones didn't reach waypoint {current_waypoint_index+1} within time limit.")
                # Force update their status anyway to prevent getting stuck
                for drone in active_drones:
                    drone.is_moving = False
                    if current_waypoint_index < len(drone.path):
                        drone.current_pose = drone.path[current_waypoint_index].copy()
                    drone.change_led_colour((255, 0, 0))  # Red for leader
                    drone.change_led_colour((0, 0, 255))  # Blue for followers
            else:
                print(f"All drones have successfully reached waypoint {current_waypoint_index+1}.")
            
            # NOW change formation AFTER reaching waypoint 1
            if current_waypoint_index == 0 and self.using_wide_formation:
                # After reaching the FIRST waypoint (index 0), switch to tighter formation
                print("\n--- FORMATION TRANSITION: Switching to tighter formation after reaching first waypoint ---")
                self.using_wide_formation = False
                
                # FIRST: Reposition drones in tighter formation at the current position
                # Get leader's current position
                leader_pos = leader.get_current_position()
                # Calculate new formation positions with tighter formation
                tight_formation_positions = self.generate_formation_positions(leader_pos, fixed_orientation)
                
                # Move followers to their new positions in tight formation
                print("Repositioning drones in tight formation before continuing...")
                active_drones = []
                
                # Leader stays in place
                # Move followers to their tight formation positions
                for i, follower in enumerate(followers):
                    if i+1 < len(tight_formation_positions):
                        pos = tight_formation_positions[i+1]  # +1 because leader position is at index 0
                        print(f"Moving {follower.drone_id} to tight formation position: {pos}")
                        follower.is_moving = True
                        follower.go_to(pos[0], pos[1], pos[2], 0.3, YawMode.FIXED_YAW, fixed_orientation, "earth", False)
                        active_drones.append(follower)
                
                # Wait for all followers to reach their positions in tight formation
                if active_drones:
                    print(f"Waiting for all {len(active_drones)} drones to reach tight formation positions...")
                    start_time = time.time()
                    max_wait = 30  # Maximum wait time in seconds
                    
                    while time.time() - start_time < max_wait and active_drones:
                        # Check each active drone
                        for drone in active_drones[:]:
                            reached_position = False
                            
                            try:
                                # Check if drone has reached its position
                                status = super(SimpleDrone, drone).get_behavior_status()
                                if status is None or status.status == BehaviorStatus.IDLE:
                                    reached_position = True
                            except Exception:
                                # Fallback to position check
                                try:
                                    current_pos = drone.position
                                    # Find this drone's target position in the formation
                                    idx = followers.index(drone)
                                    if idx+1 < len(tight_formation_positions):
                                        target_pos = tight_formation_positions[idx+1]
                                        dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                                    (current_pos[1]-target_pos[1])**2 + 
                                                    (current_pos[2]-target_pos[2])**2)
                                        if dist < 0.3:  # Within 30cm
                                            reached_position = True
                                except Exception as e:
                                    print(f"Error checking position for {drone.drone_id}: {e}")
                            
                            if reached_position:
                                print(f"{drone.drone_id} reached tight formation position.")
                                drone.is_moving = False
                                if drone == leader:
                                    drone.change_led_colour((255, 0, 0))  # Red for leader
                                else:
                                    drone.change_led_colour((0, 0, 255))  # Blue for followers
                                active_drones.remove(drone)
                        
                        # Short sleep between checks
                        if active_drones:
                            time.sleep(0.5)
                    
                    if active_drones:
                        print(f"Warning: {len(active_drones)} drones didn't reach tight formation within time limit.")
                        # Force update status to prevent getting stuck
                        for drone in active_drones:
                            drone.is_moving = False
                            drone.change_led_colour((255, 0, 0))  # Red for leader
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                    else:
                        print("All drones have successfully repositioned in tight formation.")
                    
                    # Brief pause to stabilize in formation
                    time.sleep(2.0)
                
                # SECOND: Now regenerate paths for remaining waypoints with tighter formation
                remaining_waypoints = leader_path[current_waypoint_index+1:]
                if remaining_waypoints:
                    self.generate_paths_for_all_drones(remaining_waypoints)
                    # Update drone paths to start from current position
                    for drone in all_drones:
                        drone.path = [drone.get_current_position()] + drone.path
                        drone.path_index = 0
                
                print("Continuing to next waypoint in tight formation...")
            
            # Move to next waypoint
            current_waypoint_index += 1
            
            # Brief pause between waypoints
            time.sleep(2.0)  # Increased pause between waypoints
            
        # After ALL waypoints are complete, ALWAYS transition to wider formation
        print("\n--- FORMATION TRANSITION: Moving to wider formation after completing all waypoints ---")
        self.using_wide_formation = True
        
        # Get current leader position for final formation
        leader_pos = leader.get_current_position()
        formation_positions = self.generate_formation_positions(leader_pos, fixed_orientation)
        
        # Move followers to their final positions
        active_drones = []
        for i, follower in enumerate(followers):
            if i+1 < len(formation_positions):
                pos = formation_positions[i+1]  # +1 because leader position is at index 0
                print(f"Moving {follower.drone_id} to wider formation position: {pos}")
                follower.is_moving = True
                follower.go_to(pos[0], pos[1], pos[2], 0.3, YawMode.FIXED_YAW, fixed_orientation, "earth", False)
                active_drones.append(follower)
        
        # Wait for all followers to reach their positions in wider formation
        if active_drones:
            print(f"Waiting for all {len(active_drones)} drones to reach wider formation positions...")
            start_time = time.time()
            max_wait = 60  # Maximum wait time in seconds
            
            while time.time() - start_time < max_wait and active_drones:
                # Check each active drone
                for drone in active_drones[:]:  # Use a copy for safe iteration while removing
                    reached_position = False
                    
                    try:
                        # Check if drone has reached its position
                        status = super(SimpleDrone, drone).get_behavior_status()
                        if status is None or status.status == BehaviorStatus.IDLE:
                            reached_position = True
                    except Exception:
                        # Fallback to position check
                        try:
                            current_pos = drone.position
                            idx = followers.index(drone)
                            if idx+1 < len(formation_positions):
                                target_pos = formation_positions[idx+1]
                                dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                            (current_pos[1]-target_pos[1])**2 + 
                                            (current_pos[2]-target_pos[2])**2)
                                if dist < 0.3:
                                    reached_position = True
                        except Exception as e:
                            print(f"Error checking position for {drone.drone_id}: {e}")
                    
                    if reached_position:
                        print(f"{drone.drone_id} reached wider formation position.")
                        drone.is_moving = False
                        if drone == leader:
                            drone.change_led_colour((255, 0, 0))  # Red for leader
                        else:
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                        active_drones.remove(drone)
                
                # Short sleep between checks
                if active_drones:
                    time.sleep(0.5)
            
            if active_drones:
                print(f"Warning: {len(active_drones)} drones didn't reach wider formation within time limit.")
                # Force update status to prevent getting stuck
                for drone in active_drones:
                    drone.is_moving = False
                    drone.change_led_colour((255, 0, 0))  # Red for leader
                    drone.change_led_colour((0, 0, 255))  # Blue for followers
            else:
                print("All drones have successfully positioned in wider formation before landing.")
            
            # Brief pause to stabilize in formation
            time.sleep(2.0)
        
        print("Scenario 2 mission execution completed.")


    # python3 central.py --scenario_file scenarios/scenario1_stage3.yaml

    def execute_scenario3_mission(self):
        """
        Scenario 3: Advanced 3D Obstacle avoidance with RRT*
         - Reads stage3 from YAML
         - First goes to start point in wide formation
         - Transitions to tight formation at start point
         - Computes RRT* path from start to end point
         - Flies through obstacles in tight formation
         - Continues to final point past end point
         - Transitions back to wide formation at final point
        """
        # 1) Load and validate stage3 data
        if not (self.scenario_data and "stage3" in self.scenario_data):
            print("No stage3 data loaded.")
            return
        stage3 = self.scenario_data["stage3"]
        x0, y0 = stage3["stage_center"]     # (x0, y0)
        sx_rel, sy_rel = stage3["start_point"]
        ex_rel, ey_rel = stage3["end_point"]
        obs_h = stage3["obstacle_height"]
        obs_width = stage3["obstacle_diameter"]  # Use as width and depth for box obstacles

        # 2) Compute global origin, start and end
        cruise_alt = 3.5  # Fixed altitude of 3 meters for all waypoints
        origin_wp = [0.0, 0.0, cruise_alt]
        start_wp  = [x0 - sy_rel, y0 - sx_rel, cruise_alt]
        end_wp    = [x0 - ey_rel, y0 - ex_rel, cruise_alt]
        # Create final waypoint 1.5m further in negative Y direction from end point
        final_wp  = [end_wp[0], end_wp[1] - 1.5, end_wp[2]]
        
        print("Origin:", origin_wp)
        print("Start :", start_wp)
        print("End   :", end_wp)
        print("Final :", final_wp)

        # 3) Build 3D obstacle list in global frame (box representation)
        obs_list = [
            [x0 + ox, y0 + oy, obs_h/2, obs_width, obs_width, obs_h]  # x, y, z, width, depth, height
            for ox, oy in stage3["obstacles"]
        ]
        print("3D Obstacles:", obs_list)

        # 4) Define random sampling area around the stage

        # half_x, half_y = self.scenario_data["stage_size"]
        # rand_area = [
        #     x0 - half_x/2, x0 + half_x/2,
        #     y0 - half_y/2, y0 + half_y/2,
        #     0.5, cruise_alt + 1.0  # Z range with some extra margin
        # ]

        # X span from leftmost to rightmost obstacle center + radius
        obs_xs = [o[0] for o in obs_list]
        radii  = [o[3] / 2.0 for o in obs_list]  # dx is obstacle's diameter
        min_x  = min(x - r for x, r in zip(obs_xs, radii))
        max_x  = max(x + r for x, r in zip(obs_xs, radii))

        # Y span from start to goal
        start_y, goal_y = start_wp[1], end_wp[1]
        min_y, max_y = min(start_y, goal_y), max(start_y, goal_y)

        rand_area = [
            min_x, max_x,
            min_y, max_y,
            0.0, cruise_alt
        ]

        # 5) FIRST PHASE: Go from origin to start in WIDE formation
        print("\n--- PHASE 1: Moving from origin to start point in WIDE formation ---")
        self.using_wide_formation = True
        # Only use origin and start waypoints for this phase
        initial_waypoints = [origin_wp, start_wp]
        
        # Generate paths for all drones in wide formation
        self.generate_paths_for_all_drones(initial_waypoints)

        leader = self.drones[self.leader_index]
        followers = [d for idx, d in self.drones.items() if idx != self.leader_index]
        fixed_yaw = getattr(self, "formation_orientation", 0.0)

        # Move to start point in wide formation
        for idx, wp in enumerate(initial_waypoints):
            print(f"\nMoving to waypoint {idx+1}/{len(initial_waypoints)}: {wp}")

            # Leader goes directly:
            leader.go_to(wp[0], wp[1], wp[2],
                       0.3, YawMode.FIXED_YAW, fixed_yaw,
                       "earth", False)
            leader.is_moving = True

            # Followers will have their precomputed path waypoint at same index
            for f in followers:
                if idx < len(f.path):
                    tx, ty, tz = f.path[idx]
                    f.go_to(tx, ty, tz,
                          0.3, YawMode.FIXED_YAW, fixed_yaw,
                          "earth", False)
                    f.is_moving = True

            # Wait until all have reached this index
            active = [leader] + followers
            start_t = time.time()
            while active and time.time() - start_t < 60:
                for d in active[:]:
                    arrived = False
                    # 1) behavior status
                    try:
                        st = super(SimpleDrone, d).get_behavior_status()
                        if st is None or st.status == BehaviorStatus.IDLE:
                            arrived = True
                    except:
                        pass
                    # 2) position fallback
                    if not arrived:
                        cur = d.position
                        target = d.path[idx] if d is not leader else wp
                        if math.dist(cur, target) < 0.3:
                            arrived = True
                    if arrived:
                        print(f"  {d.drone_id} reached waypoint {idx+1}")
                        d.is_moving = False
                        if d == leader:
                            d.change_led_colour((255, 0, 0))  # Red for leader
                        else:
                            d.change_led_colour((0, 0, 255))  # Blue for followers
                        active.remove(d)
                if active:
                    time.sleep(0.5)
        
        # 6) TRANSITION: Switch to tight formation at start point
        print("\n--- FORMATION TRANSITION: Switching to tight formation at start point ---")
        self.using_wide_formation = False
        
        # Get leader's current position at the start point
        leader_pos = leader.get_current_position()
        # Calculate new tight formation positions
        tight_formation_positions = self.generate_formation_positions(leader_pos, fixed_yaw)
        
        # Move followers to tight formation positions
        active_drones = []
        for i, follower in enumerate(followers):
            if i+1 < len(tight_formation_positions):
                pos = tight_formation_positions[i+1]  # +1 because leader position is at index 0
                print(f"Moving {follower.drone_id} to tight formation position: {pos}")
                follower.is_moving = True
                follower.go_to(pos[0], pos[1], pos[2], 0.3, YawMode.FIXED_YAW, fixed_yaw, "earth", False)
                active_drones.append(follower)
        
        # Wait for all followers to reach their positions in tight formation
        if active_drones:
            print(f"Waiting for {len(active_drones)} drones to reach tight formation positions...")
            start_time = time.time()
            max_wait = 30  # Maximum wait time in seconds
            
            while time.time() - start_time < max_wait and active_drones:
                for drone in active_drones[:]:
                    reached_position = False
                    try:
                        status = super(SimpleDrone, drone).get_behavior_status()
                        if status is None or status.status == BehaviorStatus.IDLE:
                            reached_position = True
                    except Exception:
                        try:
                            current_pos = drone.position
                            idx = followers.index(drone)
                            if idx+1 < len(tight_formation_positions):
                                target_pos = tight_formation_positions[idx+1]
                                dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                            (current_pos[1]-target_pos[1])**2 + 
                                            (current_pos[2]-target_pos[2])**2)
                                if dist < 0.3:  # Within 30cm
                                    reached_position = True
                        except Exception as e:
                            print(f"Error checking position for {drone.drone_id}: {e}")
                    
                    if reached_position:
                        print(f"{drone.drone_id} reached tight formation position.")
                        drone.is_moving = False
                        if drone == leader:
                            drone.change_led_colour((255, 0, 0))  # Red for leader
                        else:
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                        active_drones.remove(drone)
                
                if active_drones:
                    time.sleep(0.5)
        
        # Brief pause to stabilize in tight formation
        time.sleep(0.5)
        
        # 7) Now plan RRT* path from start to end
        print("\n--- PHASE 2: Planning RRT* path from start to end ---")
        planner = RRTStar3D(
            start=start_wp, goal=end_wp,
            obstacle_list=obs_list, rand_area=rand_area,
            expand_dist=1.0,            # Step size
            goal_sample_rate=15,        # Goal bias
            max_iter=800,               # Maximum iterations
            rewire_radius=2.5,          # Rewiring radius
            clearance=0.4               # Minimum clearance to obstacles
        )
        rrt_path = planner.plan()  # Get smoothed path
        print(f"3D RRT* path ({len(rrt_path)} pts):", rrt_path)
        
        # 6) Visualize the 3D RRT* path using matplotlib
        # Use non-interactive Agg backend to prevent Tkinter issues
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D RRT* Path Planning Visualization")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        
        # Plot obstacles as boxes
        for obs in obs_list:
            ox, oy, oz, width, depth, height = obs
            plot_cube(ax, (ox, oy, oz), (width, depth, height), color='red', alpha=0.4)
        
        # Plot start, end, final and origin points
        ax.scatter(start_wp[0], start_wp[1], start_wp[2], color='green', marker='o', s=100, label='Start')
        ax.scatter(end_wp[0], end_wp[1], end_wp[2], color='blue', marker='o', s=100, label='End')
        ax.scatter(final_wp[0], final_wp[1], final_wp[2], color='purple', marker='o', s=100, label='Final')
        ax.scatter(origin_wp[0], origin_wp[1], origin_wp[2], color='black', marker='o', s=80, label='Origin')
        
        # Plot RRT* path
        if rrt_path:
            path_x = [point[0] for point in rrt_path]
            path_y = [point[1] for point in rrt_path]
            path_z = [point[2] for point in rrt_path]
            ax.plot(path_x, path_y, path_z, color='green', linewidth=2, label='RRT* Path')
            
            # Mark waypoints with circles
            ax.scatter(path_x, path_y, path_z, color='red', marker='o', s=40, label='Waypoints')
            
            # Show line from end to final
            ax.plot([end_wp[0], final_wp[0]], [end_wp[1], final_wp[1]], [end_wp[2], final_wp[2]], 
                    'b--', linewidth=2, label='Final Path')
        
        # Set axis limits to match the visualization in the image
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-12.0, 0)  # Extended to show final point
        ax.set_zlim(0, 5.5)
        
        # Set equal aspect ratio and grid
        ax.set_box_aspect([1, 1, 0.5])  # Slightly compressed z-axis
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('3d_rrt_path_visualization.png')
        print("3D RRT* path visualization saved to '3d_rrt_path_visualization.png'")
        plt.close(fig)
        plt.close('all')
        
        # 9) THIRD PHASE: Fly RRT* path in tight formation
        print("\n--- PHASE 3: Flying RRT* path in TIGHT formation ---")
        
        # Generate paths for all drones in tight formation (maintaining self.using_wide_formation = False)
        self.generate_paths_for_all_drones(rrt_path)
        
        # Fly each waypoint in turn
        for idx, wp in enumerate(rrt_path):
            print(f"\nMoving to RRT* waypoint {idx+1}/{len(rrt_path)}: {wp}")

            # Leader goes directly:
            leader.go_to(wp[0], wp[1], wp[2],
                       0.3, YawMode.FIXED_YAW, fixed_yaw,
                       "earth", False)
            leader.is_moving = True

            # Followers will have their precomputed path waypoint at same index
            for f in followers:
                if idx < len(f.path):
                    tx, ty, tz = f.path[idx]
                    f.go_to(tx, ty, tz,
                          0.3, YawMode.FIXED_YAW, fixed_yaw,
                          "earth", False)
                    f.is_moving = True

            # Wait until all have reached this index
            active = [leader] + followers
            start_t = time.time()
            while active and time.time() - start_t < 60:
                for d in active[:]:
                    arrived = False
                    # 1) behavior status
                    try:
                        st = super(SimpleDrone, d).get_behavior_status()
                        if st is None or st.status == BehaviorStatus.IDLE:
                            arrived = True
                    except:
                        pass
                    # 2) position fallback
                    if not arrived:
                        cur = d.position
                        target = d.path[idx] if d is not leader else wp
                        if math.dist(cur, target) < 0.3:
                            arrived = True
                    if arrived:
                        print(f"  {d.drone_id} reached RRT* waypoint {idx+1}")
                        d.is_moving = False
                        if d == leader:
                            d.change_led_colour((255, 0, 0))  # Red for leader
                        else:
                            d.change_led_colour((0, 0, 255))  # Blue for followers
                        active.remove(d)
                if active:
                    time.sleep(0.5)
            
            # Add a brief pause after each waypoint is reached by all drones
            time.sleep(1.5)
        
        # 10) FOURTH PHASE: Move to final point (still in tight formation)
        print("\n--- PHASE 4: Moving to final point in TIGHT formation ---")
        
        # Generate paths for moving to final point (still in tight formation)
        final_phase_path = [end_wp, final_wp]  # From end to final
        self.generate_paths_for_all_drones(final_phase_path)
        
        # Move to final point
        print(f"\nMoving to final point: {final_wp}")
        
        # Leader goes to final point
        leader.go_to(final_wp[0], final_wp[1], final_wp[2],
                   0.3, YawMode.FIXED_YAW, fixed_yaw,
                   "earth", False)
        leader.is_moving = True
        
        # Followers move to their corresponding positions
        for f in followers:
            if len(f.path) > 0:  # Should be at least one waypoint
                tx, ty, tz = f.path[1]  # Index 1 is the final position
                f.go_to(tx, ty, tz,
                      0.3, YawMode.FIXED_YAW, fixed_yaw,
                      "earth", False)
                f.is_moving = True
        
        # Wait until all have reached the final point
        active = [leader] + followers
        start_t = time.time()
        while active and time.time() - start_t < 60:
            for d in active[:]:
                arrived = False
                # 1) behavior status
                try:
                    st = super(SimpleDrone, d).get_behavior_status()
                    if st is None or st.status == BehaviorStatus.IDLE:
                        arrived = True
                except:
                    pass
                # 2) position fallback
                if not arrived:
                    cur = d.position
                    target = d.path[1] if d is not leader else final_wp  # Index 1 is final point
                    if math.dist(cur, target) < 0.3:
                        arrived = True
                if arrived:
                    print(f"  {d.drone_id} reached final point")
                    d.is_moving = False
                    if d == leader:
                        d.change_led_colour((255, 0, 0))  # Red for leader
                    else:
                        d.change_led_colour((0, 0, 255))  # Blue for followers
                    active.remove(d)
            if active:
                time.sleep(0.5)
        
        # 11) After completing path to final point, transition to wide formation
        print("\n--- FORMATION TRANSITION: Moving to wider formation at final point ---")
        self.using_wide_formation = True
        
        # Get leader's current position at the final point
        leader_pos = leader.get_current_position()
        wide_formation_positions = self.generate_formation_positions(leader_pos, fixed_yaw)
        
        # Move followers to wide formation positions
        active_drones = []
        for i, follower in enumerate(followers):
            if i+1 < len(wide_formation_positions):
                pos = wide_formation_positions[i+1]  # +1 because leader position is at index 0
                print(f"Moving {follower.drone_id} to wider formation position: {pos}")
                follower.is_moving = True
                follower.go_to(pos[0], pos[1], pos[2], 0.3, YawMode.FIXED_YAW, fixed_yaw, "earth", False)
                active_drones.append(follower)
        
        # Wait for all followers to reach their positions in wider formation
        if active_drones:
            print(f"Waiting for all {len(active_drones)} drones to reach wider formation positions...")
            start_time = time.time()
            max_wait = 30  # Maximum wait time in seconds
            
            while time.time() - start_time < max_wait and active_drones:
                for drone in active_drones[:]:
                    reached_position = False
                    try:
                        status = super(SimpleDrone, drone).get_behavior_status()
                        if status is None or status.status == BehaviorStatus.IDLE:
                            reached_position = True
                    except Exception:
                        try:
                            current_pos = drone.position
                            idx = followers.index(drone)
                            if idx+1 < len(wide_formation_positions):
                                target_pos = wide_formation_positions[idx+1]
                                dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                            (current_pos[1]-target_pos[1])**2 + 
                                            (current_pos[2]-target_pos[2])**2)
                                if dist < 0.3:
                                    reached_position = True
                        except Exception as e:
                            print(f"Error checking position for {drone.drone_id}: {e}")
                    
                    if reached_position:
                        print(f"{drone.drone_id} reached wider formation position.")
                        drone.is_moving = False
                        if drone == leader:
                            drone.change_led_colour((255, 0, 0))  # Red for leader
                        else:
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                        active_drones.remove(drone)
                
                if active_drones:
                    time.sleep(0.5)
        
        print("Scenario 3 complete with formation transitions and 3D RRT* path planning.")

    # python3 central.py --scenario_file scenarios/scenario1_stage4.yaml

    def execute_scenario4_mission(self):
        """
        Scenario 4: Dynamic obstacle avoidance
         - Reads stage4 from YAML
         - Subscribes to dynamic obstacle positions
         - Plans and executes path from start to end while avoiding moving obstacles
        """
        # 1) Load and validate stage4 data
        if not (self.scenario_data and "stage4" in self.scenario_data):
            print("No stage4 data loaded.")
            return
        stage4 = self.scenario_data["stage4"]
        x0, y0 = stage4["stage_center"]     # (x0, y0)
        sx_rel, sy_rel = stage4["start_point"]
        ex_rel, ey_rel = stage4["end_point"]
        obs_h = stage4["obstacle_height"]
        obs_d = stage4["obstacle_diameter"]
        
        # # 2) Compute global origin, start and end
        # cruise_alt = 3.5  # Fixed altitude of 3 meters for all waypoints
        # origin_wp = [0.0, 0.0, cruise_alt]
        # start_wp = [x0 + sx_rel, y0 + sy_rel, cruise_alt]  # Convert to global coordinates
        # end_wp = [x0 + ex_rel, y0 + ey_rel, cruise_alt]    # Convert to global coordinates
        
        # print("Origin:", origin_wp)
        # print("Start :", start_wp)
        # print("End   :", end_wp)
        

        # 2) Compute global origin, start and end
        cruise_alt = 3.5  # Fixed altitude of 3 meters for all waypoints
        origin_wp = [0.0, 0.0, cruise_alt]
        start_wp  = [x0 - sy_rel, y0 - sx_rel, cruise_alt]
        start_wp[1] -= 1.5
        end_wp    = [x0 - ey_rel, y0 - ex_rel, cruise_alt]
        end_wp[1] += 2.5
        # Create final waypoint 1.5m further in negative Y direction from end point
        # final_wp  = [end_wp[0], end_wp[1] - 1.5, end_wp[2]]
        
        print("Origin:", origin_wp)
        print("Start :", start_wp)
        print("End   :", end_wp)
        # print("Final :", final_wp)

        # 3) Subscribe to dynamic obstacle positions
        self.dynamic_obstacles = {}  # Dictionary to store latest obstacle positions
        self.setup_dynamic_obstacle_subscriber()
        
        # 4) FIRST PHASE: Go from origin to start in WIDE formation
        print("\n--- PHASE 1: Moving from origin to start point in WIDE formation ---")
        self.using_wide_formation = True
        initial_waypoints = [origin_wp, start_wp]
        
        # Generate paths for all drones in wide formation
        self.generate_paths_for_all_drones(initial_waypoints)
        
        # Execute movement to start point (similar to Stage 3)
        self.execute_waypoint_movement(initial_waypoints)
        
        # 5) SECOND PHASE: Navigate through dynamic obstacles to end point
        print("\n--- PHASE 2: Navigating through dynamic obstacles to end point ---")
        self.using_wide_formation = False  # Switch to tight formation
        
        # Execute dynamic obstacle avoidance
        self.navigate_through_dynamic_obstacles(start_wp, end_wp)
        
        print("Scenario 4 complete with dynamic obstacle avoidance.")

    def setup_dynamic_obstacle_subscriber(self):
        """
        Set up a subscriber to receive dynamic obstacle positions
        """
        # Need to use a node for subscription
        try:
            # Use one of the existing drones as a node for subscription
            leader = self.drones[self.leader_index]
            
            # Create a callback to handle incoming dynamic obstacle positions
            def obstacle_position_callback(msg):
                # Extract obstacle ID from frame_id (format: "object_X")
                try:
                    obstacle_id = msg.header.frame_id
                    position = [
                        msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z
                    ]
                    # Store latest position for this obstacle
                    self.dynamic_obstacles[obstacle_id] = position
                    print(f"Received position for {obstacle_id}: {position}")
                    
                except Exception as e:
                    print(f"Error processing obstacle position: {e}")
            
            # Subscribe to the dynamic obstacles topic
            self.obstacle_sub = leader.create_subscription(
                PoseStamped,
                "/dynamic_obstacles/locations",
                obstacle_position_callback,
                10
            )
            print("Subscribed to dynamic obstacle positions")
        except Exception as e:
            print(f"Failed to set up dynamic obstacle subscriber: {e}")
            # Initialize with empty obstacles as fallback
            self.dynamic_obstacles = {}

    def execute_waypoint_movement(self, waypoints):
        """
        Execute movement through a series of waypoints for all drones
        """
        leader = self.drones[self.leader_index]
        followers = [d for idx, d in self.drones.items() if idx != self.leader_index]
        fixed_yaw = getattr(self, "formation_orientation", 0.0)
        
        for idx, wp in enumerate(waypoints):
            print(f"\nMoving to waypoint {idx+1}/{len(waypoints)}: {wp}")
            
            # Leader goes directly:
            leader.go_to(wp[0], wp[1], wp[2],
                        0.3, YawMode.FIXED_YAW, fixed_yaw,
                        "earth", False)
            leader.is_moving = True
            
            # Followers will have their precomputed path waypoint at same index
            for f in followers:
                if idx < len(f.path):
                    tx, ty, tz = f.path[idx]
                    f.go_to(tx, ty, tz,
                          0.3, YawMode.FIXED_YAW, fixed_yaw,
                          "earth", False)
                    f.is_moving = True
            
            # Wait until all have reached this index
            self.wait_for_drones_to_reach_waypoint(leader, followers, wp, idx)

    def wait_for_drones_to_reach_waypoint(self, leader, followers, wp, idx):
        """
        Wait for all drones to reach the specified waypoint
        """
        active = [leader] + followers
        start_t = time.time()
        while active and time.time() - start_t < 60:
            for d in active[:]:
                arrived = False
                # 1) behavior status
                try:
                    st = super(SimpleDrone, d).get_behavior_status()
                    if st is None or st.status == BehaviorStatus.IDLE:
                        arrived = True
                except:
                    pass
                # 2) position fallback
                if not arrived:
                    cur = d.position
                    target = d.path[idx] if d is not leader else wp
                    if math.dist(cur, target) < 0.3:
                        arrived = True
                if arrived:
                    print(f"  {d.drone_id} reached waypoint {idx+1}")
                    d.is_moving = False
                    if d == leader:
                        d.change_led_colour((255, 0, 0))  # Red for leader
                    else:
                        d.change_led_colour((0, 0, 255))  # Blue for followers
                    active.remove(d)
            if active:
                time.sleep(0.5)

    def navigate_through_dynamic_obstacles(self, start_wp, end_wp):
        """
        Navigate from start to end while avoiding dynamic obstacles
        Uses real-time obstacle position data and velocity to predict obstacle movement
        """
        leader = self.drones[self.leader_index]
        followers = [d for idx, d in self.drones.items() if idx != self.leader_index]
        fixed_yaw = getattr(self, "formation_orientation", 0.0)
        
        # Get additional parameters from scenario data
        stage4 = self.scenario_data["stage4"]
        obstacle_height = stage4["obstacle_height"]
        obstacle_diameter = stage4["obstacle_diameter"]
        safety_radius = obstacle_diameter/2 + 0.5  # Add 0.5m safety margin
        
        # Create transition to tight formation at start point
        print("\n--- FORMATION TRANSITION: Switching to tight formation at start point ---")
        leader_pos = leader.get_current_position()
        tight_formation_positions = self.generate_formation_positions(leader_pos, fixed_yaw)
        
        # Move followers to tight formation positions
        self.transition_to_formation(leader, followers, tight_formation_positions, fixed_yaw)
        
        # Brief pause to stabilize formation
        time.sleep(1.0)
        
        # Check if we're receiving obstacle data
        wait_start = time.time()
        while not self.dynamic_obstacles and time.time() - wait_start < 10.0:
            print("Waiting for dynamic obstacle data...")
            time.sleep(1.0)
        
        # If no data received, proceed with direct path
        if not self.dynamic_obstacles:
            print("Warning: No dynamic obstacle data received. Proceeding with direct path to end point.")
            waypoints = [start_wp, end_wp]
            self.execute_waypoint_movement(waypoints)
            return
        
        print(f"Received data for {len(self.dynamic_obstacles)} dynamic obstacles. Starting avoidance navigation.")
        
        # Now navigate to end point while continuously avoiding obstacles
        print("\n--- Moving to end point while avoiding dynamic obstacles ---")
        
        # Parameters for potential field navigation
        goal_attraction = 1.0
        obstacle_repulsion = 2.5
        max_velocity = 0.5  # m/s
        update_interval = 0.5  # seconds
        formation_scale = 1.0  # normal formation scale
        min_formation_scale = 0.6  # how tight the formation can get
        
        # Active navigation loop
        reached_end = False
        last_update_time = time.time()
        start_navigation_time = time.time()
        max_navigation_time = 300  # 5 minutes maximum
        
        # Variables to track obstacle velocities
        obstacle_prev_positions = {}
        obstacle_velocities = {}
        
        while not reached_end and (time.time() - start_navigation_time < max_navigation_time):
            current_time = time.time()
            dt = current_time - last_update_time
            
            # Check if we need to update path (rate limiting)
            if dt < update_interval:
                time.sleep(0.1)  # Small sleep to prevent CPU overload
                continue
                
            last_update_time = current_time
                
            # Get current positions
            leader_pos = leader.get_current_position()
            
            # Check if leader reached end point
            dist_to_goal = math.sqrt(
                (leader_pos[0]-end_wp[0])**2 + 
                (leader_pos[1]-end_wp[1])**2 + 
                (leader_pos[2]-end_wp[2])**2
            )
            
            if dist_to_goal < 0.5:  # Within 0.5m of goal
                print(f"Leader has reached end point (distance: {dist_to_goal:.2f}m)")
                reached_end = True
                continue
            
            # Handle case of no obstacle data
            if not self.dynamic_obstacles:
                # Simply move toward goal if no obstacles
                print("No obstacle data available, moving directly toward goal")
                next_pos = [
                    leader_pos[0] + (end_wp[0] - leader_pos[0]) * 0.1,  # Move 10% toward goal
                    leader_pos[1] + (end_wp[1] - leader_pos[1]) * 0.1,
                    leader_pos[2] + (end_wp[2] - leader_pos[2]) * 0.1
                ]
            else:
                # Get current obstacle positions and estimate velocities
                obstacles = []
                for obs_id, pos in self.dynamic_obstacles.items():
                    # Calculate velocity if we have previous position
                    if obs_id in obstacle_prev_positions:
                        prev_pos = obstacle_prev_positions[obs_id]
                        # Calculate velocity vector
                        vel = [
                            (pos[0] - prev_pos[0]) / dt,
                            (pos[1] - prev_pos[1]) / dt,
                            (pos[2] - prev_pos[2]) / dt
                        ]
                        obstacle_velocities[obs_id] = vel
                    else:
                        # Use default velocity if no history yet
                        obstacle_velocities[obs_id] = [0, 0, 0]
                    
                    # Store current position for next velocity calculation
                    obstacle_prev_positions[obs_id] = pos.copy()
                    
                    # Add obstacle to list with velocity information
                    obstacles.append({
                        'id': obs_id,
                        'position': pos,
                        'velocity': obstacle_velocities[obs_id],
                        'height': obstacle_height,
                        'radius': obstacle_diameter/2
                    })
                    
                # Compute potential field forces
                # 1. Goal attraction (unit vector toward goal)
                direction_to_goal = [
                    end_wp[0] - leader_pos[0],
                    end_wp[1] - leader_pos[1],
                    end_wp[2] - leader_pos[2]
                ]
                dist_to_goal = math.sqrt(sum(d*d for d in direction_to_goal))
                
                if dist_to_goal > 0.001:  # Prevent division by zero
                    attraction = [d/dist_to_goal * goal_attraction for d in direction_to_goal]
                else:
                    attraction = [0, 0, 0]
                    
                # 2. Obstacle repulsion
                repulsion = [0, 0, 0]
                closest_obstacle_dist = float('inf')
                
                for obs in obstacles:
                    obs_pos = obs['position']
                    obs_vel = obs['velocity']
                    
                    # Only consider obstacles at similar height (within obstacle height)
                    if abs(obs_pos[2] - leader_pos[2]) > obs['height']:
                        continue
                        
                    # Predict future position based on velocity (look ahead 1.5 seconds)
                    prediction_time = 1.5  # seconds
                    future_pos = [
                        obs_pos[0] + obs_vel[0] * prediction_time,
                        obs_pos[1] + obs_vel[1] * prediction_time,
                        obs_pos[2] + obs_vel[2] * prediction_time
                    ]
                    
                    # Calculate distances to both current and predicted positions
                    # Vector from current obstacle to drone
                    vec_to_drone_current = [
                        leader_pos[0] - obs_pos[0],
                        leader_pos[1] - obs_pos[1],
                        0  # Ignore Z component for horizontal avoidance
                    ]
                    
                    # Vector from predicted future obstacle position to drone
                    vec_to_drone_future = [
                        leader_pos[0] - future_pos[0],
                        leader_pos[1] - future_pos[1],
                        0  # Ignore Z component for horizontal avoidance
                    ]
                    
                    dist_current = math.sqrt(sum(d*d for d in vec_to_drone_current))
                    dist_future = math.sqrt(sum(d*d for d in vec_to_drone_future))
                    
                    # Take the minimum of current and future distances
                    dist = min(dist_current, dist_future)
                    closest_obstacle_dist = min(closest_obstacle_dist, dist)
                    
                    # Apply repulsion force with inverse square law
                    if dist < safety_radius * 3:  # Only apply when within 3x safety radius
                        # Use the vector from the closest position (current or future)
                        vec_to_use = vec_to_drone_current if dist_current <= dist_future else vec_to_drone_future
                        
                        # Normalize vector
                        if dist > 0.001:  # Prevent division by zero
                            vec_to_use = [d/dist for d in vec_to_use]
                            
                            # Force magnitude inversely proportional to distance squared
                            magnitude = obstacle_repulsion / max(0.01, (dist/safety_radius)**2)
                            
                            # Add to total repulsion
                            repulsion[0] += vec_to_use[0] * magnitude
                            repulsion[1] += vec_to_use[1] * magnitude
                            repulsion[2] += 0  # Keep at same height
                
                # 3. Compute resultant force
                resultant = [
                    attraction[0] + repulsion[0],
                    attraction[1] + repulsion[1],
                    attraction[2] + repulsion[2]
                ]
                
                # Normalize and scale to desired velocity
                resultant_mag = math.sqrt(sum(f*f for f in resultant))
                if resultant_mag > 0.001:  # Prevent division by zero
                    velocity = [f/resultant_mag * max_velocity for f in resultant]
                else:
                    velocity = [0, 0, 0]
                    
                # Calculate next position based on velocity (simple Euler integration)
                next_pos = [
                    leader_pos[0] + velocity[0] * update_interval,
                    leader_pos[1] + velocity[1] * update_interval,
                    leader_pos[2] + velocity[2] * update_interval
                ]
                
                # Adjust formation scale based on obstacle proximity
                # Tighten formation when close to obstacles
                if closest_obstacle_dist < safety_radius * 4:
                    # Scale between min_formation_scale and 1.0 based on proximity
                    formation_scale = min_formation_scale + (1.0 - min_formation_scale) * (
                        closest_obstacle_dist / (safety_radius * 4)
                    )
                    formation_scale = max(min_formation_scale, min(1.0, formation_scale))
                    print(f"Adjusting formation scale to {formation_scale:.2f} (obstacle at {closest_obstacle_dist:.2f}m)")
                else:
                    formation_scale = 1.0
            
            # Move leader to next position
            print(f"Moving leader to {next_pos} (dist to goal: {dist_to_goal:.2f}m)")
            leader.go_to(next_pos[0], next_pos[1], next_pos[2],
                       max_velocity, YawMode.FIXED_YAW, fixed_yaw,
                       "earth", False)
            leader.is_moving = True
            
            # Calculate formation positions with current scale
            formation_positions = self.generate_formation_positions(next_pos, fixed_yaw, scale=formation_scale)
            
            # Move followers to maintain formation
            for i, follower in enumerate(followers):
                if i+1 < len(formation_positions):
                    follower_pos = formation_positions[i+1]  # +1 because leader position is at index 0
                    follower.go_to(follower_pos[0], follower_pos[1], follower_pos[2],
                                 max_velocity, YawMode.FIXED_YAW, fixed_yaw,
                                 "earth", False)
                    follower.is_moving = True
            
            # Wait for a short time to let drones adjust positions
            time.sleep(update_interval)
        
        # Handle timeout case
        if not reached_end:
            print("Navigation timeout reached. Moving directly to end point.")
            waypoints = [leader.get_current_position(), end_wp]
            self.execute_waypoint_movement(waypoints)
        
        # All drones have reached the end point
        print("All drones have reached the end point successfully.")
        
        # Change to wide formation at end point
        print("\n--- FORMATION TRANSITION: Switching to wide formation at end point ---")
        self.using_wide_formation = True
        
        # Get leader's current position at the end point
        leader_pos = leader.get_current_position()
        wide_formation_positions = self.generate_formation_positions(leader_pos, fixed_yaw)
        
        # Transition to wide formation
        self.transition_to_formation(leader, followers, wide_formation_positions, fixed_yaw)

    def transition_to_formation(self, leader, followers, formation_positions, fixed_yaw):
        """
        Transition followers to specified formation positions
        """
        active_drones = []
        for i, follower in enumerate(followers):
            if i+1 < len(formation_positions):
                pos = formation_positions[i+1]  # +1 because leader position is at index 0
                print(f"Moving {follower.drone_id} to formation position: {pos}")
                follower.is_moving = True
                follower.go_to(pos[0], pos[1], pos[2], 0.3, YawMode.FIXED_YAW, fixed_yaw, "earth", False)
                active_drones.append(follower)
        
        # Wait for all followers to reach their positions
        if active_drones:
            print(f"Waiting for {len(active_drones)} drones to reach formation positions...")
            start_time = time.time()
            max_wait = 30  # Maximum wait time in seconds
            
            while time.time() - start_time < max_wait and active_drones:
                for drone in active_drones[:]:
                    reached_position = False
                    try:
                        status = super(SimpleDrone, drone).get_behavior_status()
                        if status is None or status.status == BehaviorStatus.IDLE:
                            reached_position = True
                    except Exception:
                        try:
                            current_pos = drone.position
                            idx = followers.index(drone)
                            if idx+1 < len(formation_positions):
                                target_pos = formation_positions[idx+1]
                                dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                            (current_pos[1]-target_pos[1])**2 + 
                                            (current_pos[2]-target_pos[2])**2)
                                if dist < 0.3:  # Within 30cm
                                    reached_position = True
                        except Exception as e:
                            print(f"Error checking position for {drone.drone_id}: {e}")
                    
                    if reached_position:
                        print(f"{drone.drone_id} reached formation position.")
                        drone.is_moving = False
                        if drone == leader:
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                        else:
                            drone.change_led_colour((0, 0, 255))  # Blue for followers
                        active_drones.remove(drone)
                
                if active_drones:
                    time.sleep(0.5)

    def land_all(self):
        """Land all drones simultaneously but at their current horizontal positions"""
        # Set LED color to red for all drones
        for drone in self.drones.values():
            drone.change_led_colour((255, 0, 0))  # Red for landing
        
        # Send landing commands to all drones without waiting (non-blocking)
        # Each drone will land at its current horizontal position
        print("Sending landing commands to all drones simultaneously...")
        for drone in self.drones.values():
            # Use non-blocking land (False parameter)
            drone.land(0.5, False)
        
        # Wait for all drones to reach the ground
        print("Waiting for all drones to land...")
        all_landed = False
        max_wait_time = 15  # Maximum wait time in seconds
        start_time = time.time()
        
        while not all_landed and time.time() - start_time < max_wait_time:
            all_landed = True
            for drone in self.drones.values():
                try:
                    # Check if drone has reached ground
                    current_altitude = drone.position[2]
                    if current_altitude > 0.1:  # If altitude is still significantly above ground
                        all_landed = False
                        break
                except Exception:
                    all_landed = False
                    break
        
            if not all_landed:
                time.sleep(0.2)  # Short sleep before checking again
        
        print("All drones have landed.")

    def shutdown(self):
        """Shutdown all drones in swarm"""
        for drone in self.drones.values():
            drone.shutdown()


##########################
# Main Fonksiyonu        #
##########################

def main():
    parser = argparse.ArgumentParser(description='Multi-drone formation flight mission with RRT* path planning (Scenario 2)')
    parser.add_argument('-n','--namespaces', nargs='+',
                        default=['drone0','drone1','drone2','drone3','drone4'],
                        help='Drone namespace list')
    parser.add_argument('-s','--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('--scenario_file', type=str, default='scenario3.yaml',
                        help='Path to scenario YAML file (for scenario 3)')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    rclpy.init()
    swarm = SwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time,
                           scenario_file=args.scenario_file)

    executor = MultiThreadedExecutor()
    for drone in swarm.drones.values():
        executor.add_node(drone)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        print("Starting takeoff sequence...")
        swarm.takeoff_all()
        
        
        # Determine which scenario to run based on the scenario filename
        scenario_file = args.scenario_file.lower()
        if "stage1" in scenario_file:
            print("Starting Scenario 1 mission...")
            swarm.execute_scenario1_mission()
        elif "stage2" in scenario_file:
            # Form V formation before starting the mission
            print("Forming initial V formation...")
            swarm.form_initial_formation()
            print("Starting Scenario 2 mission...")
            swarm.execute_scenario2_mission()
        elif "stage3" in scenario_file:
            # Form V formation before starting the mission
            print("Forming initial V formation...")
            swarm.form_initial_formation()
            print("Starting Scenario 3 mission...")
            swarm.execute_scenario3_mission()
        elif "stage4" in scenario_file:
            # Form V formation before starting the mission
            print("Forming initial V formation...")
            swarm.form_initial_formation()
            print("Starting Scenario 4 mission...")
            swarm.execute_scenario4_mission()
        else:
            print(f"Warning: Could not determine scenario type from filename '{args.scenario_file}'")
            print("Please specify a file containing 'stage1', 'stage2', 'stage3' or 'stage4' in the name.")
            print("Starting Scenario 3 mission by default...")
            swarm.execute_scenario3_mission()
        
        print("Landing all drones...")
        swarm.land_all()
    except KeyboardInterrupt:
        print("Operation interrupted by user")
    finally:
        print("Cleaning up...")
        # Close all matplotlib figures before shutdown
        plt.close('all')
        
        # First shutdown the executor (stop spinning)
        try:
            print("Shutting down executor...")
            executor.shutdown()
            print("Waiting for spin thread to complete...")
            spin_thread.join(timeout=2.0)
        except Exception as e:
            print(f"Error during executor shutdown: {e}")
        
        # Then shutdown the swarm interfaces
        try:
            print("Shutting down swarm interfaces...")
            swarm.shutdown()
            time.sleep(1.0)
        except Exception as e:
            print(f"Error during swarm shutdown: {e}")
        
        # Finally shutdown rclpy
        try:
            print("Shutting down rclpy...")
            rclpy.shutdown()
        except Exception as e:
            print(f"Error shutting down rclpy: {e}")
            
        print("ROS shutdown complete")
        sys.exit(0)


if __name__ == '__main__':
    main()






