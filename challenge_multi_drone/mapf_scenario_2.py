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
from mapf_utils import *


class SwarmConductor:
    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False, scenario_file: str = None):
        """
        Initialize SwarmConductor to manage multiple drones in a swarm.
        
        Args:
            drones_ns: List of drone namespaces to control
            verbose: Enable verbose output for debugging
            use_sim_time: Use simulation time instead of real time
            scenario_file: Path to YAML file containing scenario configuration
        """
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
        """
        Command all drones to take off simultaneously to a specified altitude.
        
        This function:
        1. Arms and sets all drones to offboard mode
        2. Sets LEDs to green to indicate takeoff
        3. Sends simultaneous takeoff commands to all drones
        4. Waits for all drones to reach the target altitude
        5. Adds a brief delay for stability after takeoff
        """
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
        """
        Scenario 2: Window Traversal
         - Uses waypoints calculated from the scenario file
         - Leader follows these waypoints while followers maintain formation
         - Transitions between wide and tight formations at key points:
           * Wide formation at starting point and midpoints
           * Narrow formation when traversing through windows
           * Return to original formation at the end
        """

        start_time = time.time()
        # Get current positions of all drones
        current_positions = {}
        for idx, drone in self.drones.items():
            current_positions[idx] = drone.get_current_position()
        
        # Calculate the center of the current formation
        center_x = sum(pos[0] for pos in current_positions.values()) / len(current_positions)
        center_y = sum(pos[1] for pos in current_positions.values()) / len(current_positions)
        center_z = sum(pos[2] for pos in current_positions.values()) / len(current_positions)
        
        print(f"Current center of the formation: {center_x}, {center_y}, {center_z}")

        # Store the original formation positions relative to the center
        original_formation_positions = []
        for idx, drone in self.drones.items():
            pos = drone.get_current_position()
            # Store positions relative to formation center
            rel_x = pos[0] - center_x
            rel_y = pos[1] - center_y
            original_formation_positions.append([rel_x, rel_y])

        print(f"Original formation positions relative to center: {original_formation_positions}")

        # 1. Generate waypoints for the center of formation based on window locations
        center_waypoints = []
        waypoint_types = []  # Track the type of each waypoint: "start", "window", "midpoint", "end"
        window_widths = []  # Store window widths for formation adjustments

        if self.scenario_data and "stage2" in self.scenario_data:
            stage2 = self.scenario_data["stage2"]
            y0, x0 = stage2["stage_center"]  # (y, x)
            windows = stage2["windows"]
            window_ids = sorted(windows.keys(), key=lambda k: int(k))
            offset = 3.0  # Distance before and after the windows

            # Track previous waypoint for midpoint calculation
            prev_waypoint = None
            prev_window_width = None

            for i, wid in enumerate(window_ids):
                w = windows[wid]
                wy, wx = w["center"]                  # (y, x)
                window_width = w["gap_width"]             # Window width for formation adjustment
                gz = w["distance_floor"] + w["height"] / 2.0  # Z coordinate at middle of window

                gy = y0 + wy
                gx = x0 + wx

                window_widths.append(window_width)  # Store the window width

                # Add appropriate waypoints based on window position
                if i == 0:
                    # First window: add starting point 3m before window
                    center_waypoints.append([gy, gx + offset, gz])  # Start point before window
                    waypoint_types.append("start")
                    
                    center_waypoints.append([gy, gx, gz])          # Window 1 waypoint
                    waypoint_types.append("window")
                    
                    prev_waypoint = [gy, gx, gz]
                    prev_window_width = window_width

                else:
                    current_waypoint = [gy, gx, gz]
                    
                    # Calculate and add midpoint between consecutive window waypoints
                    if prev_waypoint:
                        # Keep x and z from previous waypoint, only calculate mid y
                        mid_y = prev_waypoint[0]
                        mid_x = prev_waypoint[1] + (gx - prev_waypoint[1]) / 2
                        mid_z = prev_waypoint[2]
                        center_waypoints.append([mid_y, mid_x, mid_z])  # Add midpoint waypoint
                        waypoint_types.append("midpoint")

                        mid_y = current_waypoint[0]
                        mid_z = current_waypoint[2]
                        center_waypoints.append([mid_y, mid_x, mid_z])  # Add midpoint waypoint
                        waypoint_types.append("midpoint")

            
                    # Now add the window waypoint
                    center_waypoints.append(current_waypoint)
                    waypoint_types.append("window")
                    
                    prev_waypoint = current_waypoint  # Update previous waypoint
                    prev_window_width = window_width
                
                if i == len(window_ids) - 1:
                    center_waypoints.append([gy, gx - offset, gz])  # End point after window
                    waypoint_types.append("end")

        print(f"Generated {len(center_waypoints)} waypoints for formation center:")
        for i, wp in enumerate(center_waypoints):
            print(f"  Waypoint {i+1}: {wp} - Type: {waypoint_types[i]}")

        # 2. Calculate waypoints for all drones based on center waypoints with adjustable formations
        drone_waypoints = {idx: [] for idx in self.drones.keys()}
        fixed_orientation = 90.0  # Default formation orientation facing east


        num = 0
        # For each center waypoint, calculate corresponding position for each drone
        for i, center_wp in enumerate(center_waypoints):
            waypoint_type = waypoint_types[i]
            
            # Determine formation scale factor based on waypoint type
            scale_factor = 1.0  # Default scale (original formation)
            
            if waypoint_type == "window":
                # Get window width for this waypoint
                window_width = window_widths[num]  # Approximate mapping to window index
                num += 1
                # Calculate appropriate scale to fit through window with margin
                formation_width = max(abs(pos[0]) for pos in original_formation_positions) * 2
                scale_factor = min(1.0, (window_width * 0.2) / formation_width)
                print(f"Window {num + 1} width: {window_width}m, formation width: {formation_width}m, scale: {scale_factor}")
        

            # For each drone in the formation
            for j, drone_idx in enumerate(self.drones.keys()):
                if j < len(original_formation_positions):
                    # Get relative position of this drone in the formation
                    rel_x = original_formation_positions[j][0]
                    rel_y = original_formation_positions[j][1]
                    
                    # Apply scale factor for narrow/wide formation
                    if waypoint_type == "window":
                        # Only scale the x-coordinate (width) when passing through windows
                        scaled_rel_x = rel_x * scale_factor
                        scaled_rel_y = rel_y  # Keep y separation the same
                    else:
                        # Use original formation for start, midpoints, and end
                        scaled_rel_x = rel_x
                        scaled_rel_y = rel_y
                    
                    # Calculate absolute position for this drone
                    drone_x = center_wp[0] + scaled_rel_x
                    drone_y = center_wp[1] + scaled_rel_y
                    drone_z = center_wp[2]  # Keep same altitude as center
                    
                    # Store waypoint for this drone
                    drone_waypoints[drone_idx].append([drone_x, drone_y, drone_z])

        # 3. Move drones to waypoints one by one
        # Process each waypoint in sequence
        for wp_idx in range(len(center_waypoints)):
            print(f"\n--- Moving to waypoint {wp_idx+1}/{len(center_waypoints)} - Type: {waypoint_types[wp_idx]} ---")
            
            # Move all drones to their respective waypoints at this index
            active_drones = []
            for drone_idx, drone in self.drones.items():
                if wp_idx < len(drone_waypoints[drone_idx]):
                    target_wp = drone_waypoints[drone_idx][wp_idx]
                    print(f"Drone {drone.drone_id}: moving to {target_wp}")
                    
                    # Use non-blocking go_to call
                    drone.go_to(
                        target_wp[0], target_wp[1], target_wp[2], 
                        1.0, YawMode.FIXED_YAW, fixed_orientation,
                        "earth", False)
                    drone.is_moving = True
                    active_drones.append(drone)
            
            # Brief pause between waypoints to allow drones to move
            time.sleep(4.0)
        
        print("Scenario 2 mission execution completed.")
        end_time = time.time()
        print(f"Mission completed in {end_time - start_time:.2f} seconds")
        
    def visualize_scenario2(self, center_waypoints, drone_waypoints, waypoint_types):
        """
        Create simplified visualizations of the scenario focusing on windows and drone paths.
        Creates both a 2D (XY) plot.
        
        Args:
            center_waypoints: List of waypoints for the formation center
            drone_waypoints: Dictionary mapping drone indices to their waypoint lists
            waypoint_types: Types of each waypoint (start, window, midpoint, end)
        """
        try:
            print("Creating simplified scenario 2 visualizations...")
            
            # Create a figure for 2D plot
            plt.figure(figsize=(14, 10))
            
            # Extract coordinates for easier access
            y_points = [wp[0] for wp in center_waypoints]
            x_points = [wp[1] for wp in center_waypoints]
            z_points = [wp[2] for wp in center_waypoints]
            
            # Define colors for different drones
            drone_colors = ['#00FFFF', '#FF00FF', '#0000FF', '#FF8000', '#00FF00']
            
            # ====== Create simplified 2D (XY) plot ======
            ax = plt.gca()
            
            # Set title and labels for 2D plot
            ax.set_title('Planned Drone Paths (XY Plane)', fontsize=16, pad=20)
            ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
            ax.set_ylabel('Y (m)', fontsize=12)
            
            # 1. Plot drone paths in 2D
            for drone_idx, waypoints in drone_waypoints.items():
                color = drone_colors[drone_idx % len(drone_colors)]
                
                # Get coordinates for this drone's path
                drone_y_points = [wp[0] for wp in waypoints]
                drone_x_points = [wp[1] for wp in waypoints]
                
                # Plot the path
                ax.plot(drone_x_points, drone_y_points, 
                       color=color, linestyle='-', linewidth=2.5, 
                       marker='o', markersize=6, alpha=0.8,
                       label=f'Drone {drone_idx}')
                       
                # Mark start and end points
                if len(drone_x_points) > 0:
                    ax.plot(drone_x_points[0], drone_y_points[0], 'go', markersize=10, markeredgecolor='black')  # Start
                    ax.plot(drone_x_points[-1], drone_y_points[-1], 'ro', markersize=10, markeredgecolor='black')  # End
            
            # 2. Plot windows as gaps in walls (improved)
            if self.scenario_data and "stage2" in self.scenario_data:
                stage2 = self.scenario_data["stage2"]
                y0, x0 = stage2["stage_center"]  # (y, x)
                windows = stage2["windows"]
                window_ids = sorted(windows.keys(), key=lambda k: int(k))
                
                # Define wall parameters
                wall_length = 5  # Length of wall to draw on each side of window
                wall_thickness = 0.3  # Thickness for better visibility
                
                for wid in window_ids:
                    w = windows[wid]
                    wy, wx = w["center"]  # (y, x)
                    width = w["gap_width"]
                    
                    # Calculate window center in global coordinates
                    center_y = y0 + wy
                    center_x = x0 + wx
                    half_width = width / 2
                    
                    # Draw the entire wall first (horizontal wall with gap)
                    ax.fill_between(
                        [center_x - half_width - wall_length, center_x + half_width + wall_length],
                        [center_y - wall_thickness], [center_y + wall_thickness],
                        color='#8B4513', alpha=0.8  # Darker brown for better visibility
                    )
                    
                    # Draw the window gap with white to clearly show the opening
                    ax.fill_between(
                        [center_x - half_width, center_x + half_width],
                        [center_y - wall_thickness], [center_y + wall_thickness],
                        color='white', alpha=1.0
                    )
                    
                    # Add window number above the window
                    ax.text(center_x, center_y + 0.3, f"Window {wid}", color='red', fontsize=10, 
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2'))
            
            # Add markers for waypoint types
            for i, wp_type in enumerate(waypoint_types):
                if i < len(center_waypoints):
                    wp = center_waypoints[i]
                    x, y = wp[1], wp[0]  # x and y are reversed in the data
                    
                    if wp_type == "window":
                        # Add a star marker at window waypoints
                        ax.plot(x, y, 'y*', markersize=12, alpha=0.7)
                    elif wp_type == "start":
                        # Add text marker for start
                        ax.annotate("START", (x, y), fontsize=10, ha='center', va='bottom', 
                                  color='green', weight='bold', xytext=(0, -15), textcoords='offset points')
                    elif wp_type == "end":
                        # Add text marker for end
                        ax.annotate("END", (x, y), fontsize=10, ha='center', va='bottom', 
                                  color='red', weight='bold', xytext=(0, -15), textcoords='offset points')
            
            # Set grid and equal aspect ratio for 2D plot
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Calculate plot boundaries with margin
            margin = 2.0
            x_min, x_max = min(x_points) - margin, max(x_points) + margin
            y_min, y_max = min(y_points) - margin, max(y_points) + margin
            
            # Set axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add legend outside of the plot
            ax.legend(loc='upper left', framealpha=0.9, fontsize=10, bbox_to_anchor=(1.02, 1))
            
            # Add timestamp and notes
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(0.5, 0.01, f"Generated: {timestamp}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.94, "Mission planning visualization", ha='center', fontsize=10, style='italic')
            
            # Adjust layout for legend
            plt.subplots_adjust(right=0.85, top=0.9, bottom=0.1)
            
            # Save with tight layout
            plt.savefig('scenario2_visualization.png', dpi=300, bbox_inches='tight')
            
            # Display the plot in a non-blocking way
            plt.show(block=False)
            
            # Keep reference to avoid garbage collection
            self.vis_figure = plt.gcf()
            
            print("Simplified visualizations created and saved as 'scenario2_visualization.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def wait_for_drones(self, drones, timeout=30):
        """
        Wait for all drones to finish their movement or until timeout
        
        Args:
            drones: List of drone objects to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if all drones completed movement, False if timeout occurred
        """
        try:
            if not drones:
                print("No drones to wait for")
                return True
                
            print(f"Waiting for {len(drones)} drones to complete their movements...")
            start_time = time.time()
            
            # Track which drones we're waiting for
            waiting_for = {drone.drone_id: True for drone in drones}
            
            while time.time() - start_time < timeout:
                all_idle = True
                for drone in drones:
                    try:
                        # Check if drone is still moving
                        status = drone.get_behavior_status()
                        
                        if status is not None and status.status != BehaviorStatus.IDLE:
                            all_idle = False
                            waiting_for[drone.drone_id] = True
                        else:
                            # Drone has completed its movement
                            if drone.drone_id in waiting_for and waiting_for[drone.drone_id]:
                                print(f"Drone {drone.drone_id} completed movement")
                                waiting_for[drone.drone_id] = False
                    except Exception as e:
                        # If can't get behavior status, check position
                        try:
                            current_pos = drone.position
                            # If drone has is_moving flag, use it
                            if hasattr(drone, 'is_moving') and drone.is_moving:
                                all_idle = False
                        except Exception:
                            all_idle = False
                
                if all_idle:
                    elapsed = time.time() - start_time
                    print(f"All drones have reached their targets (took {elapsed:.2f}s)")
                    # Mark all drones as no longer moving
                    for drone in drones:
                        if hasattr(drone, 'is_moving'):
                            drone.is_moving = False
                    return True
                
                time.sleep(0.5)
            
            # Timeout reached
            print(f"Timeout ({timeout}s) reached while waiting for drones")
            # Show which drones we're still waiting for
            still_waiting = [drone_id for drone_id, waiting in waiting_for.items() if waiting]
            if still_waiting:
                print(f"Still waiting for drones: {still_waiting}")
            
            # Even if timeout reached, mark all drones as no longer moving to avoid getting stuck
            for drone in drones:
                if hasattr(drone, 'is_moving'):
                    drone.is_moving = False
            
            return False
            
        except Exception as e:
            print(f"Error in wait_for_drones: {e}")
            import traceback
            traceback.print_exc()
            
            # Even if an error occurs, mark all drones as no longer moving
            for drone in drones:
                if hasattr(drone, 'is_moving'):
                    drone.is_moving = False
                    
            return False

    def execute_scenario3_mission(self):
        """Execute scenario 3 mission (placeholder)"""
        print("Scenario 3 mission not implemented yet")
        
    def execute_scenario4_mission(self):
        """Execute scenario 4 mission (placeholder)"""
        print("Scenario 4 mission not implemented yet")

    def plan_path_to_window(self, start_center, window_center, obstacles=None):
        """
        Use RRT* to plan a path from current formation center to window approach point
        
        Args:
            start_center: Starting position [x, y, z]
            window_center: Target window position [x, y, z]
            obstacles: List of obstacles in the format [x, y, z, radius]
            
        Returns:
            List of waypoints representing the planned path
        """
        try:
            if obstacles is None:
                obstacles = self.obstacles
            
            print(f"Planning path from {start_center} to {window_center}")
            print(f"Using {len(obstacles)} obstacles for path planning")
            
            # Define planning space boundaries
            # Using a generous planning area based on start and goal positions
            min_x = min(start_center[0], window_center[0]) - 5.0
            max_x = max(start_center[0], window_center[0]) + 5.0
            min_y = min(start_center[1], window_center[1]) - 5.0
            max_y = max(start_center[1], window_center[1]) + 5.0
            min_z = min(start_center[2], window_center[2]) - 1.0
            max_z = max(start_center[2], window_center[2]) + 1.0
            
            rand_area = [min_x, max_x, min_y, max_y, min_z, max_z]
            print(f"Path planning area: {rand_area}")
            
            # For safety, if there are no obstacles or the start/goal are too close, skip complex planning
            if not obstacles or self.calculate_distance(start_center, window_center) < 3.0:
                print("Using direct path (no obstacles or short distance)")
                # Return a direct path with a midpoint to avoid obstacles
                midpoint = [
                    (start_center[0] + window_center[0]) / 2,
                    (start_center[1] + window_center[1]) / 2,
                    (start_center[2] + window_center[2]) / 2 + 0.5  # Slightly higher
                ]
                return [start_center, midpoint, window_center]
            
            # Convert obstacles to the format expected by RRTStar3D
            rrt_obstacles = []
            for obs in obstacles:
                if len(obs) >= 4:
                    x, y, z, radius = obs[:4]
                    # Convert to box obstacles (x, y, z, width, depth, height)
                    rrt_obstacles.append([x, y, z, radius*2, radius*2, radius*2])
            
            print(f"Converted {len(rrt_obstacles)} obstacles for RRT*")
            
            try:
                # Create and run the RRT* planner with reduced complexity
                rrt_star = RRTStar3D(
                    start=start_center,
                    goal=window_center,
                    obstacle_list=rrt_obstacles,
                    rand_area=rand_area,
                    expand_dist=1.0,
                    goal_sample_rate=20,
                    max_iter=100,  # Reduced for faster planning
                    rewire_radius=1.5,
                    clearance=0.5
                )
                
                path = rrt_star.plan()
                
                if path and len(path) > 0:
                    print(f"Path found with {len(path)} waypoints")
                    # Simplify the path to reduce waypoints (keep start, end, and a few key points)
                    if len(path) <= 3:
                        simplified_path = path
                    else:
                        simplified_path = [path[0]]  # Start
                        # Add a few intermediate points
                        step = max(1, len(path) // 3)
                        for i in range(step, len(path) - 1, step):
                            simplified_path.append(path[i])
                        simplified_path.append(path[-1])  # End
                    
                    print(f"Simplified path to {len(simplified_path)} waypoints")
                    return simplified_path
                else:
                    print("No path found from RRT*, using fallback direct path")
                    # Return a direct path with a midpoint as fallback
                    midpoint = [
                        (start_center[0] + window_center[0]) / 2,
                        (start_center[1] + window_center[1]) / 2,
                        (start_center[2] + window_center[2]) / 2 + 0.5  # Slightly higher
                    ]
                    return [start_center, midpoint, window_center]
            except Exception as e:
                print(f"Error in RRT* planning algorithm: {e}")
                import traceback
                traceback.print_exc()
                # Return direct path as fallback
                return [start_center, window_center]
        except Exception as e:
            print(f"Error in path planning: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal path as fallback
            return [start_center, window_center]

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two 3D points
        
        Args:
            point1: First point [x, y, z]
            point2: Second point [x, y, z]
            
        Returns:
            float: Euclidean distance between the points
        """
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2 +
            (point1[2] - point2[2]) ** 2
        )

    def move_formation_along_path(self, path, formation_positions, fixed_yaw=90.0):
        """
        Move the entire formation along a path while maintaining formation
        
        Args:
            path: List of waypoints for the formation center to follow
            formation_positions: List of current drone positions in the formation
            fixed_yaw: Fixed yaw angle (in degrees) for all drones during movement
        """
        try:
            if not path or len(path) < 2:
                print("Path too short, skipping movement")
                return
            
            print(f"Moving formation along path with {len(path)} waypoints")
            print(f"Path points: {path}")
            
            # Store current formation offsets relative to center
            formation_offsets = []
            center = [0, 0, 0]
            for pos in formation_positions:
                offset = [
                    pos[0] - center[0],
                    pos[1] - center[1],
                    0  # Keep Z offset as 0 to maintain same altitude
                ]
                formation_offsets.append(offset)
            
            print(f"Formation offsets: {formation_offsets}")
            
            # For each waypoint in the path
            for i in range(1, len(path)):
                try:
                    waypoint = path[i]
                    print(f"Moving to waypoint {i}/{len(path)-1}: {waypoint}")
                    
                    # Change LED color to cyan during path movement
                    for drone in self.drones.values():
                        drone.change_led_colour((0, 255, 255))  # Cyan for path movement
                    
                    # Move all drones to the waypoint while maintaining formation
                    active_drones = []
                    for j, (idx, drone) in enumerate(self.drones.items()):
                        try:
                            if j < len(formation_offsets):
                                offset = formation_offsets[j]
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


##########################
# Main Fonksiyonu        #
##########################

def main():
    """
    Main entry point for the multi-drone formation flight application.
    
    This function:
    1. Parses command-line arguments for drone namespaces and scenario configuration
    2. Initializes ROS2 and creates drone interfaces
    3. Sets up a multithreaded executor for concurrent drone control
    4. Executes the appropriate scenario based on the scenario filename
    5. Handles cleanup and shutdown on completion or interruption
    
    The function supports different scenarios that can be selected via the scenario_file
    parameter, looking for keywords like 'stage1', 'stage2', etc. in the filename.
    """
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






