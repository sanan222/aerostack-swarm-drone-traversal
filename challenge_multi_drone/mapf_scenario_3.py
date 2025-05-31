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
from mapf_utils import *


class SwarmConductor:
    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False, scenario_file: str = None):
        self.num_drones = len(drones_ns)
        self.drones = {}
        for idx, ns in enumerate(drones_ns):
            self.drones[idx] = SimpleDrone(ns, verbose, use_sim_time)
        self.formation_length_first_last = 0.75
        self.formation_length = 0.35  # Reduced spacing between drones (was 0.8)
        self.obstacles = []  # [x, y, z, radius]
        self.boundaries = [-10, 10, -10, 10, 0, 5]
        self.scenario_data = None
        
        print(f"Initializing SwarmConductor with {self.num_drones} drones")
        
        if scenario_file:
            try:
                print(f"Attempting to load scenario file: {scenario_file}")
                with open(scenario_file, "r") as f:
                    scenario_data = yaml.safe_load(f)
                self.scenario_data = scenario_data
                print(f"Loaded scenario data: {self.scenario_data.keys()}")
                
                if "obstacles" in scenario_data:
                    self.obstacles = scenario_data["obstacles"]
                    print(f"Loaded {len(self.obstacles)} obstacles")
                
                if "stage2" in scenario_data:
                    stage2 = scenario_data["stage2"]
                    if "windows" in stage2:
                        windows = stage2["windows"]
                        window_ids = sorted(windows.keys(), key=lambda k: int(k))
                        print(f"Loaded {len(window_ids)} windows from stage2 data")
                        for wid in window_ids:
                            print(f"  Window {wid}: {windows[wid]}")
                
                print(f"Scenario file loaded: {scenario_file}")
            except Exception as e:
                print(f"Error loading scenario file: {e}")
                import traceback
                traceback.print_exc()

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
        max_wait_time = 5  # Maximum wait time in seconds
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
        
        # After takeoff, change formation to V shape
        print("Changing formation to V shape after takeoff...")
        self.change_to_v_formation()
    
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
        max_wait_time = 5  # Maximum wait time in seconds
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
        
        
    def shutdown(self):
        """Shutdown all drone interfaces"""
        for drone in self.drones.values():
            try:
                drone.destroy_node()
            except Exception as e:
                print(f"Error shutting down drone: {e}")

    def execute_scenario1_mission(self):
        """Execute scenario 1 mission (placeholder)"""
        print("Scenario 1 mission not implemented yet")

    def execute_scenario2_mission(self):
        """Execute scenario 2 mission (placeholder)"""
        print("Scenario 2 mission not implemented yet")

    def execute_scenario3_mission(self):
        """Execute scenario 3 mission (forest traversal using obstacles as nodes)"""
        print("Starting Scenario 3: Forest Traversal Mission")
        
        """
        Scenario 3: Advanced 3D Obstacle avoidance with path planning through forest
         - Reads stage3 from YAML
         - First goes to start point in pentagon formation
         - Uses obstacles as nodes to calculate the shortest path to the end point
         - Flies through forest while maintaining pentagon formation
         - Continues to final point past end point
        """

        start_time = time.time()

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
        
        # Get current formation center
        current_positions = {}
        for idx, drone in self.drones.items():
            current_positions[idx] = drone.get_current_position()
        
        # Calculate the current center of the formation
        center_x = sum(pos[0] for pos in current_positions.values()) / len(current_positions)
        center_y = sum(pos[1] for pos in current_positions.values()) / len(current_positions)
        center_z = sum(pos[2] for pos in current_positions.values()) / len(current_positions)
        current_center = [center_x, center_y, center_z]
        print(f"Current formation center: {current_center}")
        
        # Store the current formation positions relative to the center
        current_formation = []
        for idx, drone in self.drones.items():
            pos = drone.get_current_position()
            # Store positions relative to formation center
            rel_x = pos[0] - center_x
            rel_y = pos[1] - center_y
            current_formation.append([rel_x, rel_y])
        
        print(f"Current formation relative to center: {current_formation}")
        
        # 4) Move directly to the start waypoint in pentagon formation
        print(f"Moving to start waypoint in pentagon formation: {start_wp}")
        pentagon_radius = 1.0  # Formation radius
        pentagon_formation = self.calculate_pentagon_formation(pentagon_radius, current_positions)
        
        # Move to start point in pentagon formation
        self.move_formation_to_point(start_wp, pentagon_formation)
        time.sleep(5.0)  # Small pause for stability
        
        # 5) Plan path through forest using obstacles as nodes
        print("Planning path through forest using obstacles as nodes...")
        
        # Create a graph where obstacles are nodes and calculate distances
        obstacle_graph = {}
        # Convert box obstacles back to points for path planning
        obstacle_points = [[obs[0], obs[1], cruise_alt] for obs in obs_list] # [x, y, z]
        
        # Add start and end points to the graph
        all_points = [start_wp] + obstacle_points + [end_wp]
        all_points_tuples = [tuple(p) for p in all_points]  # Convert to tuples for dict keys

        print(f"All points: {all_points}")
        
        # Build the graph where each node connects to every other node
        graph = {tuple(p): {} for p in all_points}
        
        # Add edges between all points with weights as Euclidean distances
        for i, p1 in enumerate(all_points):
            p1_tuple = tuple(p1)
            for j, p2 in enumerate(all_points):
                if i != j:  # Don't connect a point to itself
                    p2_tuple = tuple(p2)
                    # Calculate Euclidean distance as the weight
                    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
                    graph[p1_tuple][p2_tuple] = dist
        
        # Find the shortest path from start to end using Dijkstra's algorithm
        print("Finding shortest path through obstacles using Dijkstra's algorithm...")
        shortest_path = self.find_shortest_path(graph, tuple(start_wp), tuple(end_wp))
        

        print(f"Shortest path: {shortest_path}")
        # Convert path tuples back to lists for compatibility with rest of the code
        forest_path = [list(point) for point in shortest_path]
        
        # Add the final waypoint 
        forest_path.append(final_wp)
        
        print(f"Found optimal path with {len(forest_path)} waypoints")
        print(f"Path: {forest_path}")
        
        # Store all drone paths for visualization
        all_drone_paths = {}
        
        # 6) Fly through the forest following the path, using pentagon formation around obstacles
        print(f"Flying through forest following path with {len(forest_path)} waypoints")
        
        # Move through the path, adjusting the formation at each obstacle
        for i in range(1, len(forest_path)):
            waypoint = forest_path[i]
            print(f"Moving to waypoint {i}/{len(forest_path)-1}: {waypoint}")
            
            # If this is an obstacle waypoint (not start, end, or final), create a formation around it
            if i > 0 and i < len(forest_path) - 1:  # Not start or final
                print(f"Creating pentagon formation around obstacle at {waypoint}")
                
                # Determine if we need to scale the formation to avoid collisions
                # Check distance to the nearest other obstacle
                min_dist_to_other_obstacle = float('inf')
                for other_point in obstacle_points:
                    if other_point != waypoint:
                        dist = math.sqrt((waypoint[0] - other_point[0])**2 + 
                                        (waypoint[1] - other_point[1])**2)
                        min_dist_to_other_obstacle = min(min_dist_to_other_obstacle, dist)
                
                # Scale pentagon_radius based on distance to nearest obstacle
                # Don't let it get too close to other obstacles
                max_safe_radius = min_dist_to_other_obstacle * 0.4  # Use 40% of distance
                safe_radius = min(pentagon_radius, max_safe_radius)
                
                print(f"Nearest obstacle: {min_dist_to_other_obstacle}m away, using radius: {safe_radius}m")
                
                # Create pentagon formation around this obstacle
                pentagon_positions = []
                for j in range(len(self.drones)):
                    angle = j * (2 * math.pi / len(self.drones)) - math.pi/2  # Start from top
                    x = safe_radius * math.cos(angle)
                    y = safe_radius * math.sin(angle)
                    pentagon_positions.append([x, y])
                
                # Move to pentagon formation around the obstacle
                active_drones = []
                drone_paths = {}  # Store planned paths for all drones
                
                # First, plan paths for all drones
                for j, (drone_idx, drone) in enumerate(self.drones.items()):
                    if j < len(pentagon_positions):
                        offset = pentagon_positions[j]
                        # Target position for this drone
                        target_x = waypoint[0] + offset[0]
                        target_y = waypoint[1] + offset[1]
                        target_z = waypoint[2]  # Use waypoint Z for all drones
                        
                        # Current position
                        current_pos = drone.get_current_position()
                        target_pos = [target_x, target_y, target_z]
                        
                        print(f"Planning path for drone {drone_idx} from {current_pos} to {target_pos}")
                        
                        # Implement A* path planning to avoid obstacles
                        path = self.astar_plan_path(
                            start=current_pos,
                            goal=target_pos,
                            obstacles=obs_list,
                            safety_margin=0.2,  # Add 0.3m safety margin
                            bounds=[x0-10, x0+10, y0-10, y0+10, 0, 5],  # Environment bounds
                            resolution=0.3  # Grid resolution
                        )
                        
                        if not path or len(path) < 2:
                            print(f"No safe path found for drone {drone_idx}. Using direct path.")
                            # Just store direct path
                            path = [current_pos, target_pos]
                        else:
                            print(f"Found safe path with {len(path)} waypoints for drone {drone_idx}")
                        
                        # Store path for this drone
                        drone_paths[drone_idx] = path
                        
                        # Update all_drone_paths for visualization
                        if drone_idx not in all_drone_paths:
                            all_drone_paths[drone_idx] = []
                        all_drone_paths[drone_idx].extend(path)  
                        
                        active_drones.append(drone)
                
                # Now execute all drone movements simultaneously
                print(f"Executing paths for {len(active_drones)} drones simultaneously")
                
                # Determine the maximum number of waypoints across all drones
                max_waypoints = max(len(path) for path in drone_paths.values())
                
                # We'll move all drones through their paths in stages
                # For each stage, move all drones to their next waypoint
                # We'll use 4 stages maximum (start, 2 intermediate points, and goal)
                num_stages = min(4, max_waypoints)
                
                for stage in range(1, num_stages):  # Skip first waypoint (current position)
                    print(f"Moving all drones to stage {stage}/{num_stages-1}")
                    
                    # For each drone, calculate the waypoint for this stage
                    for drone_idx, drone in self.drones.items():
                        if drone_idx in drone_paths:
                            path = drone_paths[drone_idx]
                            
                            # Calculate index into this drone's path for current stage
                            # This ensures all drones reach their goal together even with different path lengths
                            if stage == num_stages - 1:
                                # Last stage - use final waypoint
                                wp_idx = len(path) - 1
                            else:
                                # Intermediate stage - evenly distribute waypoints
                                wp_idx = max(1, min(len(path) - 1, int(stage * len(path) / num_stages)))
                            
                            wp = path[wp_idx]
                            
                            print(f"Drone {drone_idx} moving to waypoint {wp_idx}/{len(path)-1}: {wp}")
                            
                            # Move to waypoint (non-blocking)
                            drone.go_to(wp[0], wp[1], wp[2], 1.0, YawMode.FIXED_YAW, 90.0, "earth", False)
                    
                    # Wait a short time for movements to start
                    time.sleep(2.0)
                    
                    # Wait for all drones to complete their movements before moving to next stage
                    # Only for intermediate stages, not the final one
                    if stage < num_stages - 1:
                        print("Waiting for all drones to reach current stage...")
                        for drone in active_drones:
                            self.wait_for_drone_to_stop(drone)
                
                # Now wait for all drones to reach their final positions
                print("Waiting for all drones to reach final positions...")
                for drone in active_drones:
                    self.wait_for_drone_to_stop(drone)
                    
                # Pause at each obstacle to "observe" it
                time.sleep(5.0)
            else:
                # For start, end, and final waypoints, just move in pentagon formation to the point
                self.move_formation_to_point(waypoint, pentagon_formation)
                
                # Record paths for visualization purposes
                for j, (drone_idx, drone) in enumerate(self.drones.items()):
                    if j < len(pentagon_formation):
                        offset = pentagon_formation[j]
                        # Target position for this drone
                        target_x = waypoint[0] + offset[0]
                        target_y = waypoint[1] + offset[1]
                        target_z = waypoint[2]  # Use waypoint Z
                        
                        # Get current position
                        current_pos = drone.get_current_position()
                        
                        # Create a simple path for visualization
                        path = [current_pos, [target_x, target_y, target_z]]
                        
                        # Update all_drone_paths
                        if drone_idx not in all_drone_paths:
                            all_drone_paths[drone_idx] = []
                        all_drone_paths[drone_idx].extend(path)
                
                time.sleep(5.0)
        
        # Set LED colors back to green after completing mission
        for drone in self.drones.values():
            drone.change_led_colour((0, 255, 0))  # Green for normal operation

        end_time = time.time()
        print(f"Mission Completion Time: {end_time - start_time:.2f} seconds")
        
        # Visualize the 3D paths and obstacles
        self.visualize_paths_and_obstacles(
            all_drone_paths, 
            obs_list, 
            start_point=start_wp, 
            end_point=end_wp, 
            final_point=final_wp
        )
        
        print("Forest traversal mission completed successfully")
    
    def visualize_paths_and_obstacles(self, drone_paths, obstacles, start_point=None, end_point=None, final_point=None):
        """
        Visualize the paths of all drones and obstacles in 3D and 2D
        
        Args:
            drone_paths: Dictionary mapping drone indices to their paths
            obstacles: List of obstacles as [x, y, z, width, depth, height]
            start_point: Start point [x, y, z]
            end_point: End point [x, y, z]
            final_point: Final point [x, y, z]
        """
        print("Generating 3D and 2D visualization of drone paths and obstacles...")
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(16, 12))
        
        # First create the 3D subplot (top)
        ax3d = fig.add_subplot(2, 1, 1, projection='3d')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('3D Visualization of Drone Paths and Obstacles')
        
        # Then create the 2D subplot (bottom)
        ax2d = fig.add_subplot(2, 1, 2)
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        ax2d.set_title('2D Top-Down View of Drone Paths and Obstacles')
        ax2d.grid(True)
        
        # Define colors for each drone
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        
        # Plot drone paths in 3D
        for drone_idx, path in drone_paths.items():
            color = colors[drone_idx % len(colors)]
            
            # Extract x, y, z coordinates from path
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            zs = [point[2] for point in path]
            
            # Plot 3D path
            ax3d.plot(xs, ys, zs, '-', color=color, linewidth=2, label=f'Drone {drone_idx}')
            ax3d.scatter(xs[0], ys[0], zs[0], color=color, marker='o', s=30)  # Start point
            ax3d.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='*', s=50)  # End point
            
            # Plot 2D path
            ax2d.plot(xs, ys, '-', color=color, linewidth=2, label=f'Drone {drone_idx}')
            ax2d.scatter(xs[0], ys[0], color=color, marker='o', s=30)  # Start point
            ax2d.scatter(xs[-1], ys[-1], color=color, marker='*', s=50)  # End point
        
        # Plot start, end, and final points in both plots
        if start_point:
            ax3d.scatter(start_point[0], start_point[1], start_point[2], color='lime', marker='s', s=100, label='Start')
            ax2d.scatter(start_point[0], start_point[1], color='lime', marker='s', s=100, label='Start')
        
        if end_point:
            ax3d.scatter(end_point[0], end_point[1], end_point[2], color='red', marker='s', s=100, label='End')
            ax2d.scatter(end_point[0], end_point[1], color='red', marker='s', s=100, label='End')
        
        if final_point:
            ax3d.scatter(final_point[0], final_point[1], final_point[2], color='black', marker='s', s=100, label='Final')
            ax2d.scatter(final_point[0], final_point[1], color='black', marker='s', s=100, label='Final')
        
        # Plot obstacles in 3D as boxes and in 2D as circles
        for obstacle in obstacles:
            x, y, z, width, depth, height = obstacle
            
            # 3D box representation
            # Lower base vertices
            z_bottom = z - height/2
            x_min = x - width/2
            x_max = x + width/2
            y_min = y - depth/2
            y_max = y + depth/2
            
            # Plot the bottom of the box
            ax3d.plot([x_min, x_max, x_max, x_min, x_min], 
                     [y_min, y_min, y_max, y_max, y_min], 
                     [z_bottom, z_bottom, z_bottom, z_bottom, z_bottom], 'k-')
            
            # Plot the top of the box
            z_top = z + height/2
            ax3d.plot([x_min, x_max, x_max, x_min, x_min], 
                     [y_min, y_min, y_max, y_max, y_min], 
                     [z_top, z_top, z_top, z_top, z_top], 'k-')
            
            # Connect the vertices
            ax3d.plot([x_min, x_min], [y_min, y_min], [z_bottom, z_top], 'k-')
            ax3d.plot([x_max, x_max], [y_min, y_min], [z_bottom, z_top], 'k-')
            ax3d.plot([x_max, x_max], [y_max, y_max], [z_bottom, z_top], 'k-')
            ax3d.plot([x_min, x_min], [y_max, y_max], [z_bottom, z_top], 'k-')
            
            # 2D circle representation
            circle = plt.Circle((x, y), width/2, fill=True, color='gray', alpha=0.5)
            ax2d.add_patch(circle)
        
        # Set equal aspect ratio for 3D plot
        # Calculate the plot bounds
        x_data = []
        y_data = []
        z_data = []
        
        # Add all path points
        for path in drone_paths.values():
            for point in path:
                x_data.append(point[0])
                y_data.append(point[1])
                z_data.append(point[2])
        
        # Add obstacle bounds
        for obstacle in obstacles:
            x, y, z, width, depth, height = obstacle
            x_data.extend([x - width/2, x + width/2])
            y_data.extend([y - depth/2, y + depth/2])
            z_data.extend([z - height/2, z + height/2])
        
        # Add start, end and final points
        for point in [start_point, end_point, final_point]:
            if point:
                x_data.append(point[0])
                y_data.append(point[1]) 
                z_data.append(point[2])
        
        # Calculate bounds
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        z_min, z_max = min(z_data), max(z_data)
        
        # Calculate center and max range
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
        
        # Set equal aspect ratio for 3D plot
        ax3d.set_xlim(x_center - max_range, x_center + max_range)
        ax3d.set_ylim(y_center - max_range, y_center + max_range)
        ax3d.set_zlim(z_center - max_range, z_center + max_range)
        
        # Set equal aspect ratio for 2D plot
        ax2d.set_xlim(x_center - max_range, x_center + max_range)
        ax2d.set_ylim(y_center - max_range, y_center + max_range)
        ax2d.set_aspect('equal')
        
        # Add legends
        ax3d.legend(loc='best')
        ax2d.legend(loc='best')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('drone_paths_visualization.png', dpi=300, bbox_inches='tight')
        
        # Show the figure (non-blocking)
        plt.draw()
        plt.pause(0.001)  # Small pause to render the figure
        print("Visualization saved to 'drone_paths_visualization.png'")
    
    def calculate_pentagon_formation(self, radius, current_positions):
        """Calculate a pentagon formation with specified radius"""
        num_drones = len(self.drones)
        
        # Define pentagon formation offsets based on number of drones
        if num_drones >= 5:
            # Full pentagon with 5 drones
            pentagon = []
            for i in range(5):
                angle = i * (2 * math.pi / 5) - math.pi/2  # Start from top
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pentagon.append([x, y])
        elif num_drones == 4:
            # 4 points of a pentagon (skip one point)
            pentagon = []
            for i in range(4):
                angle = i * (2 * math.pi / 4) - math.pi/2  # Start from top
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pentagon.append([x, y])
        elif num_drones == 3:
            # Triangle formation for 3 drones
            pentagon = []
            for i in range(3):
                angle = i * (2 * math.pi / 3) - math.pi/2  # Start from top
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pentagon.append([x, y])
        else:
            # Linear formation for fewer drones
            pentagon = []
            for i in range(num_drones):
                angle = i * (2 * math.pi / num_drones)
                pentagon.append([radius * math.cos(angle), radius * math.sin(angle)])
                
        return pentagon
    
    def astar_plan_path(self, start, goal, obstacles, safety_margin=0.3, bounds=None, resolution=0.3):
        """A* algorithm for 3D path planning
        
        Args:
            start: Start position [x, y, z]
            goal: Goal position [x, y, z]
            obstacles: List of obstacles as [x, y, z, width, depth, height]
            safety_margin: Safety margin to add around obstacles (in meters)
            bounds: Environment bounds as [xmin, xmax, ymin, ymax, zmin, zmax]
            resolution: Grid resolution for neighbor search
            
        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        class Node:
            """Node class for A* algorithm"""
            def __init__(self, position, parent=None):
                self.position = position  # [x, y, z]
                self.parent = parent
                
                self.g = 0  # Cost from start to current node
                self.h = 0  # Heuristic (estimated cost from current to goal)
                self.f = 0  # Total cost (g + h)
            
            def __eq__(self, other):
                """Check if two nodes have the same position"""
                return (abs(self.position[0] - other.position[0]) < 1e-6 and
                        abs(self.position[1] - other.position[1]) < 1e-6 and
                        abs(self.position[2] - other.position[2]) < 1e-6)

            def __hash__(self):
                """Hash function for use in sets and dictionaries"""
                return hash((round(self.position[0], 3), 
                            round(self.position[1], 3), 
                            round(self.position[2], 3)))

        # Set default bounds if not provided
        if bounds is None:
            # Set some default bounds based on start and goal
            min_x = min(start[0], goal[0]) - 5
            max_x = max(start[0], goal[0]) + 5
            min_y = min(start[1], goal[1]) - 5
            max_y = max(start[1], goal[1]) + 5
            min_z = 0
            max_z = 5
            bounds = [min_x, max_x, min_y, max_y, min_z, max_z]
        
        # Movements: 6-connected grid + diagonals (26-connected)
        movements = []
        
        # Add 6-connected movements (up, down, left, right, forward, backward)
        movements.append([resolution, 0, 0])   # Forward
        movements.append([-resolution, 0, 0])  # Backward
        movements.append([0, resolution, 0])   # Right
        movements.append([0, -resolution, 0])  # Left
        movements.append([0, 0, resolution])   # Up
        movements.append([0, 0, -resolution])  # Down
        
        # Add diagonal movements in xy-plane
        movements.append([resolution, resolution, 0])    # Forward-Right
        movements.append([resolution, -resolution, 0])   # Forward-Left
        movements.append([-resolution, resolution, 0])   # Backward-Right
        movements.append([-resolution, -resolution, 0])  # Backward-Left
        
        # Add diagonal movements with z-axis
        movements.append([resolution, 0, resolution])    # Forward-Up
        movements.append([resolution, 0, -resolution])   # Forward-Down
        movements.append([-resolution, 0, resolution])   # Backward-Up
        movements.append([-resolution, 0, -resolution])  # Backward-Down
        movements.append([0, resolution, resolution])    # Right-Up
        movements.append([0, resolution, -resolution])   # Right-Down
        movements.append([0, -resolution, resolution])   # Left-Up
        movements.append([0, -resolution, -resolution])  # Left-Down
        
        # Add 3D diagonal movements
        movements.append([resolution, resolution, resolution])       # Forward-Right-Up
        movements.append([resolution, resolution, -resolution])      # Forward-Right-Down
        movements.append([resolution, -resolution, resolution])      # Forward-Left-Up
        movements.append([resolution, -resolution, -resolution])     # Forward-Left-Down
        movements.append([-resolution, resolution, resolution])      # Backward-Right-Up
        movements.append([-resolution, resolution, -resolution])     # Backward-Right-Down
        movements.append([-resolution, -resolution, resolution])     # Backward-Left-Up
        movements.append([-resolution, -resolution, -resolution])    # Backward-Left-Down
        
        # Function to check if a position is within bounds
        def is_within_bounds(position):
            return (bounds[0] <= position[0] <= bounds[1] and
                    bounds[2] <= position[1] <= bounds[3] and
                    bounds[4] <= position[2] <= bounds[5])
        
        # Function to check if a position collides with any obstacle
        def is_collision_free(position):
            for obs in obstacles:
                ox, oy, oz, width, depth, height = obs
                # Check if position is inside obstacle box with safety margin
                half_width = width/2 + safety_margin
                half_depth = depth/2 + safety_margin
                half_height = height/2 + safety_margin
                
                if (abs(position[0] - ox) <= half_width and 
                    abs(position[1] - oy) <= half_depth and 
                    abs(position[2] - oz) <= half_height):
                    return False
            return True
        
        # Function to calculate heuristic (Euclidean distance)
        def calculate_h(position):
            return math.sqrt(
                (position[0] - goal[0]) ** 2 +
                (position[1] - goal[1]) ** 2 +
                (position[2] - goal[2]) ** 2
            )
        
        # Create start and goal nodes
        start_node = Node(start)
        goal_node = Node(goal)
        
        # Initialize open and closed sets
        open_set = {}
        closed_set = set()
        
        # Add start node to open set
        start_node_key = hash(start_node)
        open_set[start_node_key] = start_node
        
        # Set initial values for start node
        start_node.g = 0
        start_node.h = calculate_h(start)
        start_node.f = start_node.g + start_node.h
        
        # Main A* loop
        max_iterations = 5000  # Set a limit to prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Find node with lowest f value in open set
            current_key = min(open_set, key=lambda k: open_set[k].f)
            current_node = open_set[current_key]
            
            # Check if we've reached the goal (within resolution distance)
            if (math.sqrt((current_node.position[0] - goal[0]) ** 2 +
                         (current_node.position[1] - goal[1]) ** 2 +
                         (current_node.position[2] - goal[2]) ** 2) <= resolution):
                
                # Reconstruct path
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                
                # Return reversed path (start to goal)
                return path[::-1]
            
            # Move current node from open to closed set
            del open_set[current_key]
            closed_set.add(current_key)
            
            # Check all neighboring nodes
            for movement in movements:
                # Calculate new position
                new_position = [
                    current_node.position[0] + movement[0],
                    current_node.position[1] + movement[1],
                    current_node.position[2] + movement[2]
                ]
                
                # Check if new position is valid
                if not is_within_bounds(new_position) or not is_collision_free(new_position):
                    continue
                
                # Create neighbor node
                neighbor = Node(new_position, current_node)
                neighbor_key = hash(neighbor)
                
                # Skip if in closed set
                if neighbor_key in closed_set:
                    continue
                
                # Calculate g score (cost from start to neighbor)
                # Use Euclidean distance as cost
                tentative_g = current_node.g + math.sqrt(
                    movement[0] ** 2 + movement[1] ** 2 + movement[2] ** 2
                )
                
                # Check if we've found a better path to this neighbor
                if neighbor_key in open_set and tentative_g >= open_set[neighbor_key].g:
                    continue
                
                # This is the best path so far, save it
                neighbor.g = tentative_g
                neighbor.h = calculate_h(new_position)
                neighbor.f = neighbor.g + neighbor.h
                
                # Add to open set
                open_set[neighbor_key] = neighbor
        
        # No path found
        print(f"No path found after {iterations} iterations")
        return []
    
    def move_to_pentagon_formation(self, formation_positions):
        """Move drones to specified formation positions"""
        # Get current center of formation
        current_positions = {}
        for idx, drone in self.drones.items():
            current_positions[idx] = drone.get_current_position()
        
        center_x = sum(pos[0] for pos in current_positions.values()) / len(current_positions)
        center_y = sum(pos[1] for pos in current_positions.values()) / len(current_positions)
        center_z = sum(pos[2] for pos in current_positions.values()) / len(current_positions)
        
        # Move drones to new formation positions
        active_drones = []
        for idx, (drone_idx, drone) in enumerate(self.drones.items()):
            if idx < len(formation_positions):
                offset = formation_positions[idx]
                # Target position for this drone
                target_x = center_x + offset[0]
                target_y = center_y + offset[1]
                target_z = center_z  # Maintain altitude
                
                print(f"Moving drone {drone_idx} to formation position [{target_x}, {target_y}, {target_z}]")
                drone.go_to(target_x, target_y, target_z, 0.5, YawMode.FIXED_YAW, 90.0, "earth", False)
                active_drones.append(drone)
        
        # Wait for all drones to reach their positions
        if active_drones:
            print(f"Waiting for {len(active_drones)} drones to reach formation positions")
            self.wait_for_drones(active_drones, 30)
        else:
            print("Warning: No active drones to move to formation!")
    
    def move_formation_to_point(self, target_point, formation_positions):
        """Move the entire formation to a specific point while maintaining formation"""
        try:
            # Move all drones to the target point while maintaining formation
            active_drones = []
            for j, (idx, drone) in enumerate(self.drones.items()):
                try:
                    if j < len(formation_positions):
                        offset = formation_positions[j]
                        # Target position for this drone
                        target_x = target_point[0] + offset[0]
                        target_y = target_point[1] + offset[1]
                        target_z = target_point[2]  # Use target point Z for all drones
                        
                        # Move to the target position
                        print(f"Moving drone {idx} to [{target_x}, {target_y}, {target_z}]")
                        drone.go_to(target_x, target_y, target_z, 0.5, YawMode.FIXED_YAW, 90.0, "earth", False)
                        active_drones.append(drone)
                except Exception as e:
                    print(f"Error moving drone {idx} to target point: {e}")
            
            # Wait for all drones to reach their positions
            if active_drones:
                print(f"Waiting for {len(active_drones)} drones to reach target point")
                self.wait_for_drones(active_drones, 30)
            else:
                print("Warning: No active drones to move!")
                
        except Exception as e:
            print(f"Error in formation movement to point: {e}")
            import traceback
            traceback.print_exc()
    
    def find_shortest_path(self, graph, start, end):
        """Find the shortest path through a graph using Dijkstra's algorithm"""
        # Initialize distances with infinity for all nodes except start
        distances = {node: float('infinity') for node in graph}
        distances[start] = 0
        
        # Track visited nodes and previous nodes for path reconstruction
        visited = set()
        previous = {node: None for node in graph}
        
        # Main Dijkstra's algorithm loop
        while len(visited) < len(graph):
            # Find the unvisited node with minimum distance
            current = None
            min_dist = float('infinity')
            for node in graph:
                if node not in visited and distances[node] < min_dist:
                    current = node
                    min_dist = distances[node]
            
            # If no reachable nodes left or we reached the end, break
            if current is None or current == end:
                break
            
            # Mark current node as visited
            visited.add(current)
            
            # Check neighbors and update distances
            for neighbor, weight in graph[current].items():
                distance = distances[current] + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
        
        # Reconstruct the path from end to start
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        
        # Reverse the path to get start to end
        path.reverse()
        
        # If the path doesn't start with start node, no path was found
        if not path or path[0] != start:
            return []
            
        return path
    
    def move_formation_along_path(self, path, formation_positions, fixed_yaw=90.0):
        """Move the entire formation along a path while maintaining formation"""
        try:
            if not path or len(path) < 2:
                print("Path too short, skipping movement")
                return
            
            print(f"Moving formation along path with {len(path)} waypoints")
            print(f"Path points: {path}")
            
            # For each waypoint in the path
            for i in range(1, len(path)):
                try:
                    waypoint = path[i]
                    print(f"Moving to waypoint {i}/{len(path)-1}: {waypoint}")
                    
                    # Change LED color during path movement (alternating colors for visibility)
                    for j, drone in enumerate(self.drones.values()):
                        if i % 2 == 0:
                            drone.change_led_colour((0, 0, 255))  # Blue
                        else:
                            drone.change_led_colour((0, 255, 255))  # Cyan
                    
                    # Move all drones to the waypoint while maintaining formation
                    active_drones = []
                    for j, (idx, drone) in enumerate(self.drones.items()):
                        try:
                            if j < len(formation_positions):
                                offset = formation_positions[j]
                                # Target position for this drone
                                target_x = waypoint[0] + offset[0]
                                target_y = waypoint[1] + offset[1]
                                target_z = waypoint[2]  # Use waypoint Z for all drones
                                
                                # Move to the target position
                                print(f"Moving drone {idx} to [{target_x}, {target_y}, {target_z}]")
                                drone.go_to(target_x, target_y, target_z, 0.5, YawMode.FIXED_YAW, fixed_yaw, "earth", False)
                                active_drones.append(drone)
                        except Exception as e:
                            print(f"Error moving drone {idx} to waypoint: {e}")
                    
                    # Wait for all drones to reach their positions
                    if active_drones:
                        print(f"Waiting for {len(active_drones)} drones to reach waypoint {i}")
                        self.wait_for_drones(active_drones, 30)
                    else:
                        print("Warning: No active drones to move!")
                    
                    # Small delay before next waypoint
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error processing waypoint {i}: {e}")
                    continue  # Try next waypoint
            
            # Change LED color back to green after completing path
            for drone in self.drones.values():
                drone.change_led_colour((0, 255, 0))  # Green for normal operation
                
            print("Formation movement along path completed successfully")
            
        except Exception as e:
            print(f"Error in formation movement: {e}")
            import traceback
            traceback.print_exc()
    
    def wait_for_drones(self, drones, timeout=30):
        """Wait for all drones to reach their target positions or until timeout"""
        try:
            print(f"Waiting for {len(drones)} drones to reach their targets...")
            
            # Setup for timeout
            start_time = time.time()
            all_reached = False
            check_interval = 0.5  # Seconds between position checks
            
            while not all_reached and time.time() - start_time < timeout:
                all_reached = True
                
                for drone in drones:
                    if hasattr(drone, 'is_moving'):
                        # Check SimpleDrone implementation
                        if drone.is_moving:
                            all_reached = False
                            break
                    else:
                        # Check DroneInterface implementation
                        try:
                            # Get behavior status - None or IDLE means drone has stopped
                            status = drone.get_behavior_status()
                            if status is not None and status.status != BehaviorStatus.IDLE:
                                all_reached = False
                                break
                        except Exception as e:
                            print(f"Error checking drone status: {e}")
                            all_reached = False
                            break
                
                if not all_reached:
                    time.sleep(check_interval)
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print(f"Warning: Timeout reached after {elapsed:.2f} seconds while waiting for drones")
                
                # Force set all drones to not moving in case of timeout
                for drone in drones:
                    if hasattr(drone, 'is_moving'):
                        drone.is_moving = False
                
                return False
            else:
                print(f"All drones reached their targets in {elapsed:.2f} seconds")
                return True
                
        except Exception as e:
            print(f"Error waiting for drones: {e}")
            import traceback
            traceback.print_exc()
            
            # Even if an error occurs, mark all drones as no longer moving
            for drone in drones:
                if hasattr(drone, 'is_moving'):
                    drone.is_moving = False
                    
            return False

    def execute_scenario4_mission(self):
        """Execute scenario 4 mission (placeholder)"""
        print("Scenario 4 mission not implemented yet")

    def wait_for_drone_to_stop(self, drone, timeout=30):
        """Wait for a single drone to complete its movement or until timeout"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Check if drone has stopped moving
            if hasattr(drone, 'is_moving'):
                if not drone.is_moving:
                    return True
            else:
                # Alternative check for DroneInterface
                try:
                    status = drone.get_behavior_status()
                    if status is None or status.status == BehaviorStatus.IDLE:
                        return True
                except Exception as e:
                    print(f"Error checking drone status: {e}")
                    # Assume done in case of error
                    return True
            
            # Short sleep to avoid CPU spinning
            time.sleep(0.2)
        
        # If we reach here, we've timed out
        print(f"Warning: Timeout waiting for drone to complete movement")
        return False

    
##########################
# Main Fonksiyonu        #
##########################

def main():
    parser = argparse.ArgumentParser(description='Multi-drone formation flight mission with centralized MAPF (Scenario 2)')
    parser.add_argument('-n','--namespaces', nargs='+',
                        default=['drone0','drone1','drone2','drone3','drone4'],
                        help='Drone namespace list')
    parser.add_argument('-s','--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('--scenario_file', type=str, default='scenario1_stage2.yaml',
                        help='Path to scenario YAML file (for scenario 2)')
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
            print("Starting Scenario 2 mission with centralized MAPF and pentagon formation...")
            swarm.execute_scenario2_mission()
        elif "stage3" in scenario_file:
            print("Starting Scenario 3 mission...")
            swarm.execute_scenario3_mission()
        elif "stage4" in scenario_file:
            print("Starting Scenario 4 mission...")
            swarm.execute_scenario4_mission()
        else:
            print(f"Warning: Could not determine scenario type from filename '{args.scenario_file}'")
            print("Please specify a file containing 'stage1', 'stage2', 'stage3' or 'stage4' in the name.")
            print("Starting Scenario 2 mission by default...")
            swarm.execute_scenario2_mission()
        
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






