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
import yaml 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from mapf_utils import *
import heapq  # For priority queue in conflict-based search

# Define colors for different drones
DRONE_COLORS = ['red', 'blue', 'green', 'purple', 'orange']

class ConflictBasedSearch:
    """Conflict-Based Search implementation for multi-agent path finding"""
    
    @staticmethod
    def find_collision_free_paths(current_positions, target_positions, drone_radius=0.3, min_spacing=0.6):
        """Find collision-free paths from current positions to target positions
        
        Args:
            current_positions: Dictionary mapping drone IDs to current [x,y,z] positions
            target_positions: Dictionary mapping drone IDs to target [x,y,z] positions
            drone_radius: Physical radius of each drone
            min_spacing: Minimum spacing between drones (center to center)
            
        Returns:
            Dictionary mapping drone IDs to waypoints lists ([x,y,z] positions)
        """
        drone_ids = list(current_positions.keys())
        num_drones = len(drone_ids)
        
        # Group drones into batches for altitude-based separation
        num_batches = min(3, num_drones)  # Use up to 3 altitude levels
        batch_size = (num_drones + num_batches - 1) // num_batches
        batches = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_drones)
            batch_drone_ids = drone_ids[start_idx:end_idx]
            batches.append(batch_drone_ids)
        
        # Altitude offsets for each batch (in meters)
        altitude_offsets = [0.0, 0.5, 1.0]
        
        # Create paths for each drone
        paths = {}
        for batch_idx, batch in enumerate(batches):
            altitude_offset = altitude_offsets[batch_idx % len(altitude_offsets)]
            
            # Assign paths for drones in this batch
            for drone_id in batch:
                # Create waypoints with altitude offsets
                current_pos = current_positions[drone_id]
                target_pos = target_positions[drone_id]
                
                # Create a simple 3-point path with altitude change
                paths[drone_id] = [
                    # Current position
                    current_pos,
                    # Intermediate point with altitude offset
                    [current_pos[0], current_pos[1], current_pos[2] + altitude_offset],
                    # Move horizontally at offset altitude
                    [target_pos[0], target_pos[1], current_pos[2] + altitude_offset],
                    # Final position
                    target_pos
                ]
        
        # Check for potential collisions within each batch and resolve them
        for batch in batches:
            if len(batch) <= 1:
                continue  # No collision possible with single drone
                
            # Check for collisions between pairs of drones in the same batch
            for i in range(len(batch)):
                for j in range(i + 1, len(batch)):
                    drone1_id = batch[i]
                    drone2_id = batch[j]
                    
                    # Check if horizontal paths cross
                    pos1_start = paths[drone1_id][1]  # After altitude change
                    pos1_end = paths[drone1_id][2]    # Before final descent
                    pos2_start = paths[drone2_id][1]
                    pos2_end = paths[drone2_id][2]
                    
                    # Compute 2D vectors
                    vec1 = [pos1_end[0] - pos1_start[0], pos1_end[1] - pos1_start[1]]
                    vec2 = [pos2_end[0] - pos2_start[0], pos2_end[1] - pos2_start[1]]
                    
                    # Simple collision check: add a waypoint to avoid direct crossing
                    # This is a simplified approach - a full CBS would do constraint resolution
                    if (abs(vec1[0]) > 0.1 or abs(vec1[1]) > 0.1) and (abs(vec2[0]) > 0.1 or abs(vec2[1]) > 0.1):
                        # Add a detour waypoint for drone2
                        mid_x = (pos2_start[0] + pos2_end[0]) / 2
                        mid_y = (pos2_start[1] + pos2_end[1]) / 2
                        # Offset perpendicular to movement direction
                        perp_x = -vec2[1] * 0.5 / (math.sqrt(vec2[0]**2 + vec2[1]**2) + 0.001)
                        perp_y = vec2[0] * 0.5 / (math.sqrt(vec2[0]**2 + vec2[1]**2) + 0.001)
                        
                        # Insert detour waypoint
                        detour_point = [mid_x + perp_x, mid_y + perp_y, pos2_start[2]]
                        paths[drone2_id].insert(2, detour_point)
        
        return paths

class SwarmConductor:
    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False, scenario_file: str = None):
        """
        Initialize the SwarmConductor to manage a group of drones.
        
        Args:
            drones_ns: List of drone namespaces to control
            verbose: Whether to enable verbose output logging
            use_sim_time: Whether to use simulation time
            scenario_file: Path to a YAML file containing scenario configuration
                          (includes obstacles and other mission parameters)
                          
        The constructor initializes drone interfaces, sets formation parameters,
        loads scenario data if provided, and creates trajectory storage for visualization.
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
        
        # Add trajectory storage for visualization
        self.trajectories = {}
        
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

    def takeoff_all(self):
        """
        Performs simultaneous takeoff of all drones in the swarm.
        
        Coordinates the arming, offboard mode setting, and takeoff sequence for all drones.
        The function sends non-blocking takeoff commands to all drones simultaneously and then
        monitors their progress until they reach the target altitude or the maximum wait time expires.
        An additional delay is added at the end for stability.
        """
        # First arm and set to offboard mode for all drones
        for drone in self.drones.values():
            drone.arm()
            drone.offboard()
        
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

    def create_formation_comparison_plot(self, formation_results):
        """Create a bar chart comparing formation completion times"""
        formations = list(formation_results.keys())
        times = [formation_results[f] for f in formations]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use orange color for all bars
        bars = ax.bar(formations, times, color='orange')
        
        # Add labels and title with larger font sizes
        ax.set_xlabel('Formation Type', fontsize=14)
        ax.set_ylabel('Transition Time (seconds)', fontsize=14)
        ax.set_title('Formation Transition Performance Comparison', fontsize=16, fontweight='bold')
        
        # Set y-axis to start slightly below the minimum time
        min_time = min(times)
        ax.set_ylim(bottom=max(0, min_time * 0.95))
        
        # Add values on top of bars with larger font size
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f} s', ha='center', va='bottom', fontsize=12)
        
        # Increase tick font size
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('formation_comparison.png', dpi=300)
        
        return fig, ax

    def get_formation_positions(self, formation_type, num_drones):
        """Generate formation positions for different formation types"""
        formation_positions = []
        formation_length = 0.75  # Distance between drones
        
        if formation_type == "line":
            # Create a horizontal line formation (all drones in a row)
            for i in range(num_drones):
                # Place drones in a horizontal line with equal spacing (x axis)
                formation_positions.append([i * formation_length - (num_drones - 1) * formation_length / 2, 0])
                
        elif formation_type == "v":
            # Use V formation
            formation_positions = Choreographer.v_formation(
                length=formation_length,
                angle_deg=30.0,
                orientation_deg=90,  # V points forward
                center=[0, 0],
                num_drones=num_drones
            )
            
        elif formation_type == "square":
            # Create a square formation as shown in the image
            side_length = formation_length
            
            if num_drones <= 4:
                # For 4 or fewer drones, place one on each corner
                corners = [
                    [-side_length/2, -side_length/2],  # Bottom-left
                    [side_length/2, -side_length/2],   # Bottom-right
                    [side_length/2, side_length/2],    # Top-right
                    [-side_length/2, side_length/2]    # Top-left
                ]
                formation_positions = corners[:num_drones]
            else:
                # For 5 drones (like in the image), place 4 at corners and 1 on the right side
                formation_positions = [
                    [-side_length/2, side_length/2],    # Top-left
                    [side_length/2, side_length/2],     # Top-right
                    [side_length/2, -side_length/2],    # Bottom-right
                    [-side_length/2, -side_length/2],   # Bottom-left
                    [side_length/2, 0]                  # Middle of the right side
                ]
                
                # If more than 5 drones, add additional ones
                if num_drones > 5:
                    # Place remaining drones at other side midpoints
                    additional_positions = [
                        [0, side_length/2],             # Middle of the top side
                        [-side_length/2, 0],            # Middle of the left side
                        [0, -side_length/2],            # Middle of the bottom side
                        [0, 0]                          # Center
                    ]
                    
                    for i in range(num_drones - 5):
                        if i < len(additional_positions):
                            formation_positions.append(additional_positions[i])
        
        elif formation_type == "orbit":
            # Create a circular/orbit formation
            radius = formation_length
            for i in range(num_drones):
                angle = 2 * math.pi * i / num_drones
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                formation_positions.append([x, y])
        
        elif formation_type == "grid":
            # Create a grid formation
            # Calculate the grid dimensions
            grid_size = math.ceil(math.sqrt(num_drones))
            grid_spacing = formation_length
            
            # Create grid positions
            for i in range(num_drones):
                row = i // grid_size
                col = i % grid_size
                x = (col - (grid_size - 1) / 2) * grid_spacing
                y = (row - (grid_size - 1) / 2) * grid_spacing
                formation_positions.append([x, y])
        
        elif formation_type == "staggered":
            # Create a staggered formation in an M-like shape
            # For 5 drones, create an M shape pattern
            side_length = formation_length  # Add this line to define side_length
            if num_drones == 5:
                # Pattern for M shape
                formation_positions = [
                    [-side_length, side_length],             # Top left
                    [-side_length/2, 0],                     # Middle left
                    [0, side_length],                        # Middle top
                    [side_length/2, 0],                      # Middle right
                    [side_length, side_length]               # Top right
                ]
            else:
                # For other numbers of drones, create a more general M-like pattern
                spacing = formation_length
                positions = []
                
                if num_drones >= 2:
                    # Left leg of M
                    positions.append([-spacing, spacing])    # Top left
                    positions.append([-spacing, 0])          # Middle left
                
                if num_drones >= 3:
                    # Middle peak of M
                    positions.append([0, spacing])           # Middle top
                
                if num_drones >= 5:
                    # Right leg of M
                    positions.append([spacing, 0])           # Middle right
                    positions.append([spacing, spacing])     # Top right
                
                # Add any additional drones to complete the M shape or extend it
                additional_positions = [
                    [-spacing, -spacing],                    # Bottom left
                    [0, 0],                                  # Center
                    [spacing, -spacing],                     # Bottom right
                    [0, -spacing],                           # Bottom center
                ]
                
                remaining = num_drones - len(positions)
                for i in range(remaining):
                    if i < len(additional_positions):
                        positions.append(additional_positions[i])
                
                formation_positions = positions
        
        elif formation_type == "columnN":
            # Create a column formation (drones stacked front to back)
            for i in range(num_drones):
                # Place drones in a single front-to-back column with equal spacing
                formation_positions.append([i * formation_length - (num_drones - 1) * formation_length / 2, 0])
        
        elif formation_type == "free":
            # Random formation within constraints
            max_radius = formation_length * 1.5
            for i in range(num_drones):
                while True:
                    # Generate random position within a circle
                    r = max_radius * math.sqrt(random.random())
                    theta = random.random() * 2 * math.pi
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    
                    # Check minimum distance from other drones
                    min_dist = float('inf')
                    for pos in formation_positions:
                        dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                        min_dist = min(min_dist, dist)
                    
                    # Accept position if it's not too close to others
                    if len(formation_positions) == 0 or min_dist >= formation_length * 0.6:
                        formation_positions.append([x, y])
                        break
        
        else:  # Default: use line formation
            for i in range(num_drones):
                formation_positions.append([i * formation_length - (num_drones - 1) * formation_length / 2, 0])
        
        return formation_positions

    def transition_formation_safely(self, leader_position, orientation, formation_type):
        """
        Safely transition the drone swarm to a new formation using altitude-based separation.
        
        This function implements a collision-free formation change by:
        1. Getting current positions of all drones
        2. Generating new formation positions based on the requested formation type
        3. Calculating target positions using the leader position and orientation
        4. Computing collision-free paths using the ConflictBasedSearch algorithm
        5. Executing the computed paths with altitude-based separation
        
        Args:
            leader_position: 3D position [x,y,z] that serves as the reference point for the formation
            orientation: Orientation angle in degrees for the formation (relative to the global frame)
            formation_type: Type of formation to transition to (e.g., "line", "v", "square", etc.)
            
        Returns:
            The formation positions relative to the leader (used for subsequent movement)
        """
        print(f"Performing safe formation transition to {formation_type}")
        
        # Get current positions of all drones
        current_positions = {}
        for idx, drone in self.drones.items():
            current_positions[idx] = drone.get_current_position()
        
        # Generate new formation positions
        formation_positions = self.get_formation_positions(formation_type, len(self.drones))
        
        # Calculate target positions for each drone
        target_positions = {}
        for idx, drone in enumerate(self.drones.items()):
            drone_idx, _ = drone
            if idx < len(formation_positions):
                # Get formation position for this drone
                form_pos = formation_positions[idx]
                
                # Calculate rotated position based on orientation angle
                rot_angle = math.radians(orientation)
                rotated_x = form_pos[0] * math.cos(rot_angle) - form_pos[1] * math.sin(rot_angle)
                rotated_y = form_pos[0] * math.sin(rot_angle) + form_pos[1] * math.cos(rot_angle)
                
                # Calculate final position
                new_x = leader_position[0] + rotated_x
                new_y = leader_position[1] + rotated_y
                new_z = leader_position[2]
                
                target_positions[drone_idx] = [new_x, new_y, new_z]
        
        # Calculate safe paths using conflict-based search
        safe_paths = ConflictBasedSearch.find_collision_free_paths(
            current_positions, target_positions, drone_radius=0.3, min_spacing=0.6
        )
        
        # Execute the safe paths
        print("Executing safe formation transition...")
        
        # Parameters for go_to
        speed = 0.7  # Slower speed for safety
        yaw_mode = YawMode.FIXED_YAW
        frame_id = "earth"
        
        # Execute each waypoint of the paths for all drones
        for waypoint_idx in range(4):  # Maximum 4 waypoints per path
            # Check if any drone has this waypoint
            active_drones = False
            for drone_idx, path in safe_paths.items():
                if waypoint_idx < len(path):
                    active_drones = True
                    break
            
            if not active_drones:
                break
                
            # Send all drones to their next waypoint simultaneously
            for drone_idx, path in safe_paths.items():
                if waypoint_idx < len(path):
                    waypoint = path[waypoint_idx]
                    drone = self.drones[drone_idx]
                    print(f"Moving drone {drone_idx} to waypoint {waypoint_idx}: {waypoint}")
                    
                    # Use non-blocking go_to
                    drone.go_to(waypoint[0], waypoint[1], waypoint[2], 
                             speed, yaw_mode, orientation, frame_id, False)
            
            # Wait for all drones to reach their waypoints
            all_reached = False
            while not all_reached:
                all_reached = True
                for drone_idx, path in safe_paths.items():
                    if waypoint_idx < len(path):
                        drone = self.drones[drone_idx]
                        pos = drone.get_current_position()
                        wp = path[waypoint_idx]
                        
                        dist = math.sqrt(
                            (pos[0] - wp[0])**2 +
                            (pos[1] - wp[1])**2 +
                            (pos[2] - wp[2])**2
                        )
                        
                        if dist > 0.3:  # Threshold for waypoint completion
                            all_reached = False
                            break
                
                if not all_reached:
                    time.sleep(0.1)
            
            print(f"All drones reached waypoint {waypoint_idx}")
        
        print(f"Formation transition to {formation_type} completed safely")
        return formation_positions

    def execute_scenario1_mission(self):
        """Execute scenario 1 mission with changing formations during a single circular motion"""
        print("Executing Scenario 1 mission with changing formations during circular motion...")
        
        # List of formations to test
        formations = ["line", "v", "square", "orbit", "grid", "staggered", "columnN", "free"]
        formation_results = {}
        
        # Initialize trajectories dict
        self.trajectories = {idx: [] for idx in self.drones.keys()}
        
        # Get current positions of all drones
        current_positions = {}
        for idx, drone in self.drones.items():
            current_positions[idx] = drone.get_current_position()
            # Add initial position to trajectory
            self.trajectories[idx].append(current_positions[idx])
        
        # Calculate the center of the current formation
        center_x = sum(pos[0] for pos in current_positions.values()) / len(current_positions)
        center_y = sum(pos[1] for pos in current_positions.values()) / len(current_positions)
        center_z = sum(pos[2] for pos in current_positions.values()) / len(current_positions)
        
        print(f"Current center of the formation: {center_x}, {center_y}, {center_z}")
        
        # Define circle parameters
        circle_radius = 2.0  # Reduced radius (was 3.0) to make circle smaller
        circle_center = [0.0, 0.0, 1.5]  # Center of the circle
        
        # Calculate number of waypoints per formation
        # We'll divide the circle into segments for each formation
        segments_per_formation = 4  # Number of segments to complete before changing formations
        total_segments = segments_per_formation * len(formations)
        angle_per_segment = 360 / total_segments
        
        print(f"Circle divided into {total_segments} segments ({segments_per_formation} segments per formation)")
        
        # Define parameters for go_to
        speed = 0.75  # Reduced speed (was 1.0) for safer movement
        yaw_mode = YawMode.PATH_FACING
        yaw_angle = None
        frame_id = "earth"
        
        # Record mission start time
        mission_start_time = time.time()
        
        # Execute the circular motion with changing formations
        for formation_idx, formation_type in enumerate(formations):
            print(f"\n\n===== FORMATION: {formation_type.upper()} =====\n")
            
            # Calculate start angle for this formation segment
            start_angle = formation_idx * segments_per_formation * angle_per_segment
            
            # Record formation transition start time
            formation_start_time = time.time()
            
            # Calculate first position for this formation
            start_rad = math.radians(start_angle)
            start_x = circle_center[0] + circle_radius * math.cos(start_rad)
            start_y = circle_center[1] + circle_radius * math.sin(start_rad)
            start_z = circle_center[2]
            
            # Calculate the formation orientation angle (tangent to circle)
            start_orientation_angle = start_angle + 90  # Tangent to circle
            
            # Leader position for this formation segment
            leader_position = [start_x, start_y, start_z]
            
            # Transition to the new formation safely
            if formation_idx > 0:  # Skip for first formation
                print(f"Transitioning to {formation_type} formation...")
                formation_positions = self.transition_formation_safely(
                    leader_position, start_orientation_angle, formation_type
                )
            else:
                # For the first formation, generate positions and move directly
                formation_positions = self.get_formation_positions(formation_type, len(self.drones))
                
                # Calculate target positions for each drone
                target_positions = {}
                for idx, drone in enumerate(self.drones.items()):
                    drone_idx, _ = drone
                    if idx < len(formation_positions):
                        # Get formation position for this drone
                        form_pos = formation_positions[idx]
                        
                        # Calculate rotated position based on orientation angle
                        rot_angle = math.radians(start_orientation_angle)
                        rotated_x = form_pos[0] * math.cos(rot_angle) - form_pos[1] * math.sin(rot_angle)
                        rotated_y = form_pos[0] * math.sin(rot_angle) + form_pos[1] * math.cos(rot_angle)
                        
                        # Calculate final position
                        new_x = start_x + rotated_x
                        new_y = start_y + rotated_y
                        new_z = start_z
                        
                        target_positions[drone_idx] = [new_x, new_y, new_z]
                
                # Move all drones to their positions
                for idx, drone in self.drones.items():
                    pos = target_positions[idx]
                    print(f"Moving drone {idx} to initial position: {pos}")
                    drone.go_to(pos[0], pos[1], pos[2], 
                             speed, YawMode.FIXED_YAW, start_orientation_angle, frame_id, False)
                
                # Wait for all drones to reach their positions
                all_reached = False
                while not all_reached:
                    all_reached = True
                    for idx, drone in self.drones.items():
                        pos = drone.get_current_position()
                        target_pos = target_positions[idx]
                        
                        dist = math.sqrt(
                            (pos[0] - target_pos[0])**2 +
                            (pos[1] - target_pos[1])**2 +
                            (pos[2] - target_pos[2])**2
                        )
                        
                        if dist > 0.5:
                            all_reached = False
                            break
                    
                    if not all_reached:
                        time.sleep(0.1)
                
                # Give extra time for stabilization in first formation
                time.sleep(1.0)
            
            # Record formation transition completion time
            formation_transition_time = time.time() - formation_start_time
            
            # Now move through the segments for this formation
            segment_start_time = time.time()
            
            for segment in range(segments_per_formation):
                current_angle = start_angle + segment * angle_per_segment
                
                # Calculate position on the circle for this angle
                rad = math.radians(current_angle)
                x = circle_center[0] + circle_radius * math.cos(rad)
                y = circle_center[1] + circle_radius * math.sin(rad)
                z = circle_center[2]
                
                # Calculate the formation orientation angle (tangent to circle)
                orientation_angle = current_angle + 90  # Tangent to circle
                
                print(f"Moving to position at angle {current_angle} degrees with orientation {orientation_angle} degrees")
                
                # Generate formation positions for each drone
                for idx, drone in enumerate(self.drones.items()):
                    drone_idx, drone_obj = drone
                    
                    # Get formation position for this drone
                    if idx < len(formation_positions):
                        form_pos = formation_positions[idx]
                        
                        # Calculate rotated position based on orientation angle
                        rot_angle = math.radians(orientation_angle)
                        rotated_x = form_pos[0] * math.cos(rot_angle) - form_pos[1] * math.sin(rot_angle)
                        rotated_y = form_pos[0] * math.sin(rot_angle) + form_pos[1] * math.cos(rot_angle)
                        
                        # Calculate final position
                        new_x = x + rotated_x
                        new_y = y + rotated_y
                        
                        print(f"Sending drone {drone_idx} to [{new_x:.2f}, {new_y:.2f}, {z:.2f}]")
                        
                        # Use non-blocking go_to (False for wait_for_ready)
                        drone_obj.go_to(new_x, new_y, z, speed, yaw_mode, yaw_angle, frame_id, False)
                
                # Wait for all drones to reach their waypoints
                all_reached = False
                while not all_reached:
                    all_reached = True
                    for idx, drone in enumerate(self.drones.items()):
                        drone_idx, drone_obj = drone
                        
                        if idx < len(formation_positions):
                            # Calculate target position for this drone
                            form_pos = formation_positions[idx]
                            rot_angle = math.radians(orientation_angle)
                            rotated_x = form_pos[0] * math.cos(rot_angle) - form_pos[1] * math.sin(rot_angle)
                            rotated_y = form_pos[0] * math.sin(rot_angle) + form_pos[1] * math.cos(rot_angle)
                            target_x = x + rotated_x
                            target_y = y + rotated_y
                            target_z = z
                            
                            # Calculate distance to target
                            pos = drone_obj.get_current_position()
                            dist = math.sqrt(
                                (pos[0] - target_x)**2 +
                                (pos[1] - target_y)**2 +
                                (pos[2] - target_z)**2
                            )
                            
                            if dist > 0.5:  # Reduced threshold for faster transitions
                                all_reached = False
                                print(f"Drone {drone_idx} is {dist:.2f} m from target")
                                break
                    
                    if not all_reached:
                        time.sleep(0.1)
            
            # Calculate time spent flying in this formation (excluding transition)
            formation_flight_time = time.time() - segment_start_time
            
            # Store total time (transition + flight)
            formation_results[formation_type] = formation_transition_time
            
            print(f"{formation_type.upper()} formation:")
            print(f"  Transition time: {formation_transition_time:.2f} seconds")
            print(f"  Flight time: {formation_flight_time:.2f} seconds")
            print(f"  Total segment time: {formation_transition_time + formation_flight_time:.2f} seconds")
        
        # Record total mission completion time
        total_mission_time = time.time() - mission_start_time
        
        # Print comparison of results
        print("\n\n===== FORMATION TRANSITION COMPARISON =====")
        for formation, transition_time in formation_results.items():
            print(f"{formation.upper()} formation transition: {transition_time:.2f} seconds")
        
        # Determine fastest formation transition
        fastest_formation = min(formation_results, key=formation_results.get)
        print(f"\nFastest formation transition: {fastest_formation.upper()} ({formation_results[fastest_formation]:.2f} seconds)")
        print(f"\nTotal mission completion time: {total_mission_time:.2f} seconds")
        
        # Create a comparison plot of all formations
        self.create_formation_comparison_plot(formation_results)
        
        print("\nScenario 1 with formation transitions completed")

    def execute_scenario2_mission(self):
        """Execute scenario 2 mission (placeholder)"""
        print("Scenario 2 mission not implemented yet")

    def execute_scenario3_mission(self):
        """Execute scenario 3 mission (placeholder)"""
        print("Scenario 3 mission not implemented yet")
        
    def execute_scenario4_mission(self):
        """Execute scenario 4 mission (placeholder)"""
        print("Scenario 4 mission not implemented yet")


# Main Function
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






