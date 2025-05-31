#!/usr/bin/env python3

import argparse
import sys
import time
import random
import threading
import math
from math import radians, cos, sin, atan2
from typing import List
import copy

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
        self.leader_index = 0  # Default leader is drone0
        
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

        # Set different LED colors for different drones to distinguish them
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255)    # Purple
        ]
        
        for i, drone in enumerate(self.drones.values()):
            color_idx = i % len(colors)
            drone.change_led_colour(colors[color_idx])
        
        # Send takeoff commands to all drones without waiting (non-blocking)
        print("Sending takeoff commands to all drones simultaneously...")
        for drone in self.drones.values():
            # Use non-blocking takeoff (False parameter)
            drone.takeoff(2.0, 0.8, False)
        
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
                    if current_altitude < 1.9:  # Adjusted from 1.4 to 1.9 for higher altitude
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
        
        # After takeoff, change formation to pentagon shape
        print("Changing formation to pentagon shape after takeoff...")
        self.change_to_pentagon_formation()
        
    def change_to_pentagon_formation(self):
        """
        Change the formation to a pentagon shape after takeoff
        """
        print("Setting up pentagon formation...")
        
        # Set an orientation for the formation (we'll use this throughout the mission)
        formation_orientation = 90.0  # Exactly forward-facing (positive Y direction)
        self.formation_orientation = formation_orientation
        
        # Get all drones
        all_drones = list(self.drones.values())
        
        # Calculate the current center of the formation
        current_positions = [drone.get_current_position() for drone in all_drones]
        formation_center = [
            sum(pos[0] for pos in current_positions) / len(current_positions),
            sum(pos[1] for pos in current_positions) / len(current_positions),
            sum(pos[2] for pos in current_positions) / len(current_positions)
        ]
        
        # Generate pentagon formation positions
        pentagon_positions = self.generate_formation_positions(formation_center, formation_orientation)
        
        # Move all drones to the pentagon formation positions
        print("Moving drones to pentagon formation positions...")
        self.transition_to_formation(all_drones, pentagon_positions, formation_orientation)
        
        # Set different LED colors for different drones to distinguish them
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255)    # Purple
        ]
        
        for i, drone in enumerate(all_drones):
            color_idx = i % len(colors)
            drone.change_led_colour(colors[color_idx])
            
        print("Pentagon formation complete.")

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
        """Execute scenario 3 mission (placeholder)"""
        print("Scenario 3 mission not implemented yet")

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
        

        # 2) Compute global origin, start and end
        cruise_alt = 3.5  # Fixed altitude of 4.5 meters for all waypoints (increased from 3.5)
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

        start_time = time.time()
        
        # 4) FIRST PHASE: Go from origin to start in pentagon formation
        print("\n--- PHASE 1: Moving from origin to start point in pentagon formation ---")
        self.using_wide_formation = False
        
        # Set an orientation for the formation (looking forward)
        forward_orientation = 90.0  # Exactly forward-facing (positive Y direction)
        self.formation_orientation = forward_orientation
        
        # Get all drones
        all_drones = list(self.drones.values())
        
        # Define the current center (origin) and target (start point)
        current_positions = [drone.get_current_position() for drone in all_drones]
        formation_center = [
            sum(pos[0] for pos in current_positions) / len(current_positions),
            sum(pos[1] for pos in current_positions) / len(current_positions),
            sum(pos[2] for pos in current_positions) / len(current_positions)
        ]
        
        # Generate pentagon formation positions at the origin
        origin_pentagon_positions = self.generate_formation_positions(origin_wp, forward_orientation, scale=0.8)
        
        # Move all drones to the pentagon formation at origin
        print("Moving to pentagon formation at origin...")
        self.transition_to_formation(all_drones, origin_pentagon_positions, forward_orientation)
        
        # Brief pause to stabilize formation
        time.sleep(1.0)
        
        # Generate pentagon formation positions at the start point
        start_pentagon_positions = self.generate_formation_positions(start_wp, forward_orientation, scale=0.8)
        
        # # Move the formation to the start point while maintaining the pentagon shape
        # print("Moving pentagon formation to start point...")
        # self.move_formation_to_waypoint(start_pentagon_positions, forward_orientation)
        
        # print("Formation has reached start point in pentagon formation.")
        
        # 5) SECOND PHASE: Navigate through dynamic obstacles to end point
        print("\n--- PHASE 2: Navigating through dynamic obstacles to end point ---")
        self.using_wide_formation = False  # Switch to tight formation
        
        # Execute dynamic obstacle avoidance
        self.navigate_through_dynamic_obstacles(start_wp, end_wp)
        
        print("Scenario 4 complete with dynamic obstacle avoidance.")

        end_time = time.time()
        print(f"Mission Completion Time: {end_time - start_time:.2f} seconds")

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
                    # print(f"Received position for {obstacle_id}: {position}")
                    
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
        Modified to treat all drones equally in a pentagon formation
        """
        all_drones = list(self.drones.values())
        # Always use forward orientation (90 degrees)
        fixed_yaw = 90.0
        self.formation_orientation = fixed_yaw
        
        for idx, wp in enumerate(waypoints):
            print(f"\nMoving to waypoint {idx+1}/{len(waypoints)}: {wp}")
            
            # All drones move to their respective path points
            for drone in all_drones:
                if idx < len(drone.path):
                    tx, ty, tz = drone.path[idx]
                    drone.go_to(tx, ty, tz,
                              0.8, YawMode.FIXED_YAW, fixed_yaw,
                              "earth", False)
                    drone.is_moving = True
            
            # Wait until all have reached this index
            self.wait_for_all_drones_to_reach_waypoint(all_drones, idx)

    def wait_for_all_drones_to_reach_waypoint(self, drones, idx):
        """
        Wait for all drones to reach the specified waypoint index
        """
        active = drones.copy()
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
                    try:
                        cur = d.position
                        target = d.path[idx]
                        if math.dist(cur, target) < 0.3:
                            arrived = True
                    except Exception as e:
                        print(f"Error checking position for {d.drone_id}: {e}")
                
                if arrived:
                    print(f"  {d.drone_id} reached waypoint {idx+1}")
                    d.is_moving = False
                    d.change_led_colour((0, 0, 255))  # Blue for all drones
                    active.remove(d)
            if active:
                time.sleep(0.5)

    def navigate_through_dynamic_obstacles(self, start_wp, end_wp):
        """
        Navigate from start to end with a hybrid approach:
        1. Use A* for planning the formation center path to avoid obstacles
        2. Use potential field for real-time obstacle avoidance
        
        Modified to plan for the center of a pentagon formation while adaptively resizing
        the formation based on obstacle proximity
        """
        # Ensure the formation is always pointing forward (90 degrees)
        fixed_yaw = 90.0
        self.formation_orientation = fixed_yaw
        
        # Get additional parameters from scenario data
        stage4 = self.scenario_data["stage4"]
        obstacle_height = stage4["obstacle_height"]
        obstacle_diameter = stage4["obstacle_diameter"]
        safety_radius = obstacle_diameter/2 + 0.5  # Add 0.5m safety margin
        
        # Get all drones
        all_drones = list(self.drones.values())
        
        # Create transition to pentagon formation at start point
        print("\n--- FORMATION TRANSITION: Switching to pentagon formation at start point ---")
        
        # Calculate current formation center
        current_positions = [drone.get_current_position() for drone in all_drones]
        formation_center = [
            sum(pos[0] for pos in current_positions) / len(current_positions),
            sum(pos[1] for pos in current_positions) / len(current_positions),
            sum(pos[2] for pos in current_positions) / len(current_positions)
        ]
        
        # Generate pentagon formation positions around the center point
        pentagon_positions = self.generate_formation_positions(formation_center, fixed_yaw)
        
        # Move drones to tight pentagon formation positions
        self.transition_to_formation(all_drones, pentagon_positions, fixed_yaw)
        
        # Brief pause to stabilize formation
        time.sleep(1.0)
        
        # Check if we're receiving obstacle data
        wait_start = time.time()
        while not self.dynamic_obstacles and time.time() - wait_start < 10.0:
            print("Waiting for dynamic obstacle data...")
            time.sleep(1.0)
        
        # Initialize variables to track obstacle velocities
        obstacle_prev_positions = {}
        obstacle_velocities = {}
        last_velocity_update = time.time()
        
        # Prepare obstacles for path planning
        planning_obstacles = []
        prediction_time = 0.0  # No prediction for initial planning
        
        # Wait to get some velocity data
        print("Collecting initial dynamic obstacle data...")
        collection_start = time.time()
        while time.time() - collection_start < 2.0:
            # Update obstacle velocities
            current_time = time.time()
            dt = current_time - last_velocity_update
            
            if dt > 0.5:  # Update velocities every 0.5 seconds
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
                
                last_velocity_update = current_time
            
            time.sleep(0.1)
        
        # Now get the current position of formation center as planning start point
        current_positions = [drone.get_current_position() for drone in all_drones]
        formation_center = [
            sum(pos[0] for pos in current_positions) / len(current_positions),
            sum(pos[1] for pos in current_positions) / len(current_positions),
            sum(pos[2] for pos in current_positions) / len(current_positions)
        ]
        
        # If no obstacle data, proceed with direct path
        if not self.dynamic_obstacles:
            print("Warning: No dynamic obstacle data received. Proceeding with direct path to end point.")
            waypoints = [formation_center, end_wp]
            
            # Calculate the formation center at each waypoint
            for wp in waypoints:
                # Generate formation positions around waypoint
                formation_positions = self.generate_formation_positions(wp, fixed_yaw)
                
                # Move all drones in formation
                self.move_formation_to_waypoint(formation_positions, fixed_yaw)
            
            return
        
        print(f"Received data for {len(self.dynamic_obstacles)} dynamic obstacles. Starting hybrid navigation.")
        
        # Parameters for hybrid navigation
        goal_attraction = 1.0
        obstacle_repulsion = 3.0
        max_velocity = 0.5  # m/s
        update_interval = 0.75  # seconds
        formation_scale = 1.0  # normal formation scale
        min_formation_scale = 0.5  # minimum formation scale
        planning_horizon = 1.0  # meters, for A* planning
        replan_distance = 0.2  # Distance to travel before replanning
        replan_time = 0.5  # Time interval for replanning (seconds)
        
        # Variables for A* path planning
        current_path = []
        current_path_index = 0
        last_planning_time = 0
        last_planning_position = formation_center
        
        # Main navigation loop
        reached_end = False
        last_update_time = time.time()
        start_navigation_time = time.time()
        max_navigation_time = 300  # 5 minutes maximum
        
        while not reached_end and (time.time() - start_navigation_time < max_navigation_time):
            current_time = time.time()
            dt = current_time - last_update_time
            
            # Check if we need to update (rate limiting)
            if dt < update_interval:
                time.sleep(0.05)  # Small sleep to prevent CPU overload
                continue
                
            last_update_time = current_time
            
            # Get current formation center by averaging all drone positions
            current_positions = [drone.get_current_position() for drone in all_drones]
            formation_center = [
                sum(pos[0] for pos in current_positions) / len(current_positions),
                sum(pos[1] for pos in current_positions) / len(current_positions),
                sum(pos[2] for pos in current_positions) / len(current_positions)
            ]
            
            # Check if formation center reached end point
            dist_to_goal = math.sqrt(
                (formation_center[0]-end_wp[0])**2 + 
                (formation_center[1]-end_wp[1])**2 + 
                (formation_center[2]-end_wp[2])**2
            )
            
            if dist_to_goal < 0.5:  # Within 0.5m of goal
                print(f"Formation has reached end point (distance: {dist_to_goal:.2f}m)")
                reached_end = True
                continue
            
            # Update obstacle velocities
            if current_time - last_velocity_update > 0.5:  # Update velocities every 0.5 seconds
                for obs_id, pos in self.dynamic_obstacles.items():
                    # Calculate velocity if we have previous position
                    if obs_id in obstacle_prev_positions:
                        prev_pos = obstacle_prev_positions[obs_id]
                        dt_velocity = current_time - last_velocity_update
                        # Calculate velocity vector
                        vel = [
                            (pos[0] - prev_pos[0]) / dt_velocity,
                            (pos[1] - prev_pos[1]) / dt_velocity,
                            (pos[2] - prev_pos[2]) / dt_velocity
                        ]
                        # Apply exponential smoothing for velocity (alpha = 0.7)
                        if obs_id in obstacle_velocities:
                            old_vel = obstacle_velocities[obs_id]
                            smoothed_vel = [
                                0.7 * vel[0] + 0.3 * old_vel[0],
                                0.7 * vel[1] + 0.3 * old_vel[1],
                                0.7 * vel[2] + 0.3 * old_vel[2]
                            ]
                            obstacle_velocities[obs_id] = smoothed_vel
                        else:
                            obstacle_velocities[obs_id] = vel
                    
                    # Store current position for next velocity calculation
                    obstacle_prev_positions[obs_id] = pos.copy()
                
                last_velocity_update = current_time
            
            # Check if we need to replan the path
            # Conditions for replanning:
            # 1. No current path OR
            # 2. Reached end of current path OR
            # 3. Time since last planning > replan_time OR
            # 4. Distance moved since last planning > replan_distance
            distance_since_planning = math.sqrt(
                (formation_center[0] - last_planning_position[0])**2 +
                (formation_center[1] - last_planning_position[1])**2 +
                (formation_center[2] - last_planning_position[2])**2
            )
            
            time_since_planning = current_time - last_planning_time
            
            need_replan = (
                not current_path or 
                current_path_index >= len(current_path) - 1 or
                time_since_planning > replan_time or
                distance_since_planning > replan_distance
            )
            
            if need_replan:
                # Convert dynamic obstacles to a format suitable for A* planning
                # Use velocity prediction to anticipate obstacle positions
                planning_obstacles = []
                prediction_time = 1.0  # Look ahead 1 second for planning
                
                for obs_id, pos in self.dynamic_obstacles.items():
                    # Get velocity
                    vel = obstacle_velocities.get(obs_id, [0, 0, 0])
                    
                    # Predict future position
                    future_pos = [
                        pos[0] + vel[0] * prediction_time,
                        pos[1] + vel[1] * prediction_time,
                        pos[2] + vel[2] * prediction_time
                    ]
                    
                    # Add both current and predicted positions as obstacles
                    # This creates a "tube" of avoidance along the predicted path
                    planning_obstacles.append([pos[0], pos[1], pos[2], obstacle_diameter/2 + 0.3])  # Current position
                    planning_obstacles.append([future_pos[0], future_pos[1], future_pos[2], obstacle_diameter/2 + 0.2])  # Future position
                    
                    # Add additional sampling points along predicted path for continuous coverage
                    for t in [0.25, 0.5, 0.75]:
                        interp_pos = [
                            pos[0] + vel[0] * prediction_time * t,
                            pos[1] + vel[1] * prediction_time * t,
                            pos[2] + vel[2] * prediction_time * t
                        ]
                        planning_obstacles.append([interp_pos[0], interp_pos[1], interp_pos[2], obstacle_diameter/2 + 0.2])
                
                # Set environment bounds for planning - using larger bounds for better paths
                planning_bounds = [
                    min(formation_center[0], end_wp[0]) - 10,
                    max(formation_center[0], end_wp[0]) + 10,
                    min(formation_center[1], end_wp[1]) - 10,
                    max(formation_center[1], end_wp[1]) + 10,
                    min(formation_center[2], end_wp[2]) - 0.7,
                    max(formation_center[2], end_wp[2]) + 0.7
                ]
                
                # Plan path using A* from current formation center
                print(f"Planning path from {formation_center} to {end_wp} with A*, obstacles: {len(planning_obstacles)}")
                current_path = self.astar_plan_path(
                    formation_center, end_wp, planning_obstacles, 
                    planning_distance=planning_horizon, 
                    resolution=0.2,  # Using slightly larger resolution for faster planning
                    safety_margin=0.3,
                    bounds=planning_bounds
                )
                
                # If A* planning was successful
                if current_path and len(current_path) > 1:
                    print(f"A* planning successful - found path with {len(current_path)} waypoints")

                else:
                    # If A* failed, use direct path with vertical offset
                    print("A* planning failed, using direct path with vertical offset")
                    
                    # Calculate vector to goal
                    to_goal = [end_wp[0] - formation_center[0], end_wp[1] - formation_center[1]]
                    dist_to_goal = math.sqrt(to_goal[0]**2 + to_goal[1]**2)
                    
                    # Normalize direction
                    if dist_to_goal > 0.001:
                        to_goal = [to_goal[0]/dist_to_goal, to_goal[1]/dist_to_goal]
                    else:
                        to_goal = [0, 0]
                    
                    # Create intermediate waypoints with increasing height
                    current_path = [formation_center]
                    
                    # Add a higher point to fly over obstacles
                    mid_point1 = [
                        formation_center[0] + to_goal[0] * min(dist_to_goal/2, planning_horizon/2),
                        formation_center[1] + to_goal[1] * min(dist_to_goal/2, planning_horizon/2),
                        formation_center[2] + 0.4  # Add altitude
                    ]
                    current_path.append(mid_point1)
                    
                    # Add another point closer to goal
                    mid_point2 = [
                        formation_center[0] + to_goal[0] * min(dist_to_goal*0.75, planning_horizon*0.75),
                        formation_center[1] + to_goal[1] * min(dist_to_goal*0.75, planning_horizon*0.75),
                        formation_center[2] + 0.25  # Slightly lower than mid_point1
                    ]
                    current_path.append(mid_point2)
                
                # Update planning tracking variables
                current_path_index = 0
                last_planning_time = current_time
                last_planning_position = formation_center.copy()
                
                print(f"Using path with {len(current_path)} waypoints")
            
            # Get next waypoint from the planned path
            if current_path and current_path_index < len(current_path):
                next_waypoint = current_path[current_path_index]
                
                # Calculate distance to next waypoint
                dist_to_waypoint = math.sqrt(
                    (formation_center[0] - next_waypoint[0])**2 +
                    (formation_center[1] - next_waypoint[1])**2 +
                    (formation_center[2] - next_waypoint[2])**2
                )
                
                # Check if we've reached this waypoint
                if dist_to_waypoint < 0.2:  # Within 20cm of waypoint
                    current_path_index += 1
                    print(f"Reached waypoint {current_path_index}/{len(current_path)}")
                    
                    # If we've reached the last waypoint, just use the goal
                    if current_path_index >= len(current_path):
                        next_waypoint = end_wp
                    else:
                        next_waypoint = current_path[current_path_index]
            else:
                # If we don't have a path, use the goal directly
                next_waypoint = end_wp
            
            # Now apply potential field forces for real-time obstacle avoidance
            # 1. Attraction to next waypoint (simplified, already have waypoint)
            attraction_point = next_waypoint
            
            # 2. Repulsive forces from obstacles
            current_obstacles = []
            closest_obstacle_dist = float('inf')
            
            for obs_id, pos in self.dynamic_obstacles.items():
                # Only consider obstacles at similar height
                if abs(pos[2] - formation_center[2]) > obstacle_height:
                    continue
                    
                # Get velocity of this obstacle
                vel = obstacle_velocities.get(obs_id, [0, 0, 0])
                
                # Predict future position (look ahead 1.5 seconds for avoidance)
                avoidance_prediction_time = 1.5
                future_pos = [
                    pos[0] + vel[0] * avoidance_prediction_time,
                    pos[1] + vel[1] * avoidance_prediction_time,
                    pos[2] + vel[2] * avoidance_prediction_time
                ]
                
                # Calculate distances to both current and predicted positions
                dist_current = math.sqrt(
                    (formation_center[0] - pos[0])**2 +
                    (formation_center[1] - pos[1])**2
                )
                
                dist_future = math.sqrt(
                    (formation_center[0] - future_pos[0])**2 +
                    (formation_center[1] - future_pos[1])**2
                )
                
                # Use the smaller distance for safety
                min_dist = min(dist_current, dist_future)
                closest_obstacle_dist = min(closest_obstacle_dist, min_dist)
                
                # Add obstacle to current obstacles list
                current_obstacles.append({
                    'id': obs_id,
                    'position': pos,
                    'future_position': future_pos,
                    'velocity': vel,
                    'distance': min_dist
                })
            
            # Calculate repulsion vector
            repulsion = [0, 0, 0]
            
            # Define a critical distance threshold where potential fields take over
            critical_distance = safety_radius * 2.5  # When obstacles get this close, potential fields dominate
            close_obstacles_exist = False
            
            for obs in current_obstacles:
                # Check if any obstacle is getting critically close
                if obs['distance'] < critical_distance:
                    close_obstacles_exist = True
                
                # Apply repulsion with increasing strength as obstacles get closer
                if obs['distance'] < safety_radius * 3.0:
                    # Use current or future position based on which is closer
                    pos = obs['position']
                    future_pos = obs['future_position']
                    
                    dist_current = math.sqrt(
                        (formation_center[0] - pos[0])**2 +
                        (formation_center[1] - pos[1])**2
                    )
                    
                    dist_future = math.sqrt(
                        (formation_center[0] - future_pos[0])**2 +
                        (formation_center[1] - future_pos[1])**2
                    )
                    
                    use_pos = pos if dist_current <= dist_future else future_pos
                    use_dist = min(dist_current, dist_future)
                    
                    # Calculate repulsion direction (away from obstacle)
                    direction = [
                        formation_center[0] - use_pos[0],
                        formation_center[1] - use_pos[1],
                        0  # Keep at same height
                    ]
                    
                    # Normalize direction
                    mag = math.sqrt(direction[0]**2 + direction[1]**2)
                    if mag > 0.001:
                        direction = [d/mag for d in direction]
                        
                        # Calculate repulsion factor - stronger as obstacles get closer
                        # Exponential increase as we get closer to obstacle
                        distance_factor = (safety_radius * 3) / max(0.1, use_dist)
                        repulsion_factor = obstacle_repulsion * (distance_factor ** 2)
                        
                        # Cap maximum repulsion to avoid extreme forces
                        repulsion_factor = min(8.0, repulsion_factor)
                        
                        # Add to total repulsion
                        repulsion[0] += direction[0] * repulsion_factor
                        repulsion[1] += direction[1] * repulsion_factor
            
            # Determine whether to follow A* path or use pure potential fields
            # based on obstacle proximity
            if close_obstacles_exist:
                print(f"Obstacle getting close ({closest_obstacle_dist:.2f}m) - using potential fields for avoidance")
                
                # Use direct attraction to goal with strong repulsion
                direction_to_goal = [
                    end_wp[0] - formation_center[0],
                    end_wp[1] - formation_center[1],
                    end_wp[2] - formation_center[2]
                ]
                
                # Normalize
                mag = math.sqrt(sum(d*d for d in direction_to_goal))
                if mag > 0.001:
                    direction_to_goal = [d/mag * goal_attraction for d in direction_to_goal]
                
                # Combine with strong repulsion but maintain forward movement
                # Increase the weight of the goal attraction to ensure forward progress
                movement_vector = [
                    direction_to_goal[0] * 1.5 + repulsion[0],
                    direction_to_goal[1] * 1.5 + repulsion[1],
                    direction_to_goal[2]  # Keep Z from waypoint planning
                ]

                min_forward_component = 0.5* max_velocity
                if math.sqrt(movement_vector[0]**2 + movement_vector[1]**2) < min_forward_component:
                    # Calculate direction to goal
                    to_goal = [end_wp[0] - formation_center[0], end_wp[1] - formation_center[1]]
                    goal_dist = math.sqrt(to_goal[0]**2 + to_goal[1]**2)
                    
                    if goal_dist > 0.001:
                        # Normalize direction to goal
                        to_goal = [to_goal[0]/goal_dist, to_goal[1]/goal_dist]
                        
                        # Add minimum forward component in the direction of the goal
                        movement_vector[0] += to_goal[0] * min_forward_component
                        movement_vector[1] += to_goal[1] * min_forward_component
                
                # Normalize and scale to max velocity
                mag = math.sqrt(sum(d*d for d in movement_vector))
                if mag > 0.001:
                    movement_vector = [d/mag * max_velocity for d in movement_vector]
                
                # Force replanning after obstacle avoidance maneuver
                # But don't set to zero to avoid constant replanning
                # Instead, reduce the time until next replan
                last_planning_time = current_time - (replan_time * 0.8)
            else:

                print("No obstacles, following A* path")
                # Normal operation - follow A* path with gentle repulsion for adjustments
                movement_vector = [
                    next_waypoint[0] - formation_center[0],
                    next_waypoint[1] - formation_center[1],
                    next_waypoint[2] - formation_center[2]
                ]
                
            
            # Calculate next formation center
            next_center = [
                formation_center[0] + movement_vector[0] * update_interval,
                formation_center[1] + movement_vector[1] * update_interval,
                formation_center[2] + movement_vector[2] * update_interval
            ]
            
            # Adjust formation scale based on obstacle proximity
            if closest_obstacle_dist < safety_radius * 3:
                # Scale between min_formation_scale and 1.0 based on proximity
                formation_scale = min_formation_scale + (1.0 - min_formation_scale) * (
                    closest_obstacle_dist / (safety_radius * 3)
                )
                formation_scale = max(min_formation_scale, min(1.0, formation_scale))
                print(f"Adjusting formation scale to {formation_scale:.2f} (obstacle at {closest_obstacle_dist:.2f}m)")
            else:
                formation_scale = 1.0
            
            # Calculate new formation positions based on the next center position
            print(f"Moving formation to: {next_center} (dist to goal: {dist_to_goal:.2f}m, formation scale: {formation_scale:.2f})")
            new_formation_positions = self.generate_formation_positions(next_center, fixed_yaw, scale=formation_scale)
            
            # Move all drones to maintain formation
            for i, drone in enumerate(all_drones):
                if i < len(new_formation_positions):
                    drone_pos = new_formation_positions[i]
                    drone.go_to(drone_pos[0], drone_pos[1], drone_pos[2],
                              max_velocity, YawMode.FIXED_YAW, fixed_yaw,
                              "earth", False)
                    drone.is_moving = True
            
            # Wait for a short time to let drones adjust positions
            time.sleep(update_interval)
        
        # Handle timeout case
        if not reached_end:
            print("Navigation timeout reached. Moving directly to end point.")
            
            # Calculate the formation positions at the end waypoint
            end_formation_positions = self.generate_formation_positions(end_wp, fixed_yaw)
            
            # Move all drones to the end formation
            self.move_formation_to_waypoint(end_formation_positions, fixed_yaw)
        
        # All drones have reached the end point
        print("All drones have reached the end point successfully.")
        
        # Change to wide formation at end point
        print("\n--- FORMATION TRANSITION: Switching to wide formation at end point ---")
        self.using_wide_formation = True
        
        # Calculate current formation center
        current_positions = [drone.get_current_position() for drone in all_drones]
        formation_center = [
            sum(pos[0] for pos in current_positions) / len(current_positions),
            sum(pos[1] for pos in current_positions) / len(current_positions),
            sum(pos[2] for pos in current_positions) / len(current_positions)
        ]
        
        # Generate wide pentagon formation positions
        wide_formation_positions = self.generate_formation_positions(formation_center, fixed_yaw)
        
        # Move drones to wide formation
        self.move_formation_to_waypoint(wide_formation_positions, fixed_yaw)

    def transition_to_formation(self, drones, formation_positions, fixed_yaw=None):
        """
        Transition drones to specified formation positions
        
        Args:
            drones: List of drone objects to move
            formation_positions: List of positions for each drone
            fixed_yaw: The fixed yaw angle for the drones
        """
        # Ensure formation is facing forward if yaw not specified
        if fixed_yaw is None:
            fixed_yaw = 90.0
            self.formation_orientation = fixed_yaw
        
        active_drones = []
        for i, drone in enumerate(drones):
            if i < len(formation_positions):
                pos = formation_positions[i]
                print(f"Moving {drone.drone_id} to formation position: {pos}")
                drone.is_moving = True
                drone.go_to(pos[0], pos[1], pos[2], 0.7, YawMode.FIXED_YAW, fixed_yaw, "earth", False)
                active_drones.append(drone)
        
        # Wait for all drones to reach their positions
        if active_drones:
            print(f"Waiting for {len(active_drones)} drones to reach formation positions...")
            start_time = time.time()
            max_wait = 30  # Maximum wait time in seconds (reduced from 60)
            
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
                            idx = drones.index(drone)
                            if idx < len(formation_positions):
                                target_pos = formation_positions[idx]
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
                        drone.change_led_colour((0, 0, 255))  # Blue for all drones in formation
                        active_drones.remove(drone)
                
                if active_drones:
                    time.sleep(0.5)
                    
    def move_formation_to_waypoint(self, formation_positions, fixed_yaw=None):
        """
        Move all drones to the specified formation positions
        
        Args:
            formation_positions: List of positions for each drone
            fixed_yaw: The fixed yaw angle for the drones
        """
        # Ensure formation is facing forward if yaw not specified
        if fixed_yaw is None:
            fixed_yaw = 90.0
            self.formation_orientation = fixed_yaw
            
        all_drones = list(self.drones.values())
        active_drones = []
        
        # Send all drones to their target positions
        for i, drone in enumerate(all_drones):
            if i < len(formation_positions):
                pos = formation_positions[i]
                print(f"Moving {drone.drone_id} to waypoint formation position: {pos}")
                drone.is_moving = True
                drone.go_to(pos[0], pos[1], pos[2], 0.8, YawMode.FIXED_YAW, fixed_yaw, "earth", False)
                active_drones.append(drone)
        
        # Wait for all drones to reach their positions
        if active_drones:
            print(f"Waiting for {len(active_drones)} drones to reach waypoint...")
            start_time = time.time()
            max_wait = 30  # Maximum wait time in seconds (reduced from 60)
            
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
                            idx = all_drones.index(drone)
                            if idx < len(formation_positions):
                                target_pos = formation_positions[idx]
                                dist = math.sqrt((current_pos[0]-target_pos[0])**2 + 
                                            (current_pos[1]-target_pos[1])**2 + 
                                            (current_pos[2]-target_pos[2])**2)
                                if dist < 0.3:  # Within 30cm
                                    reached_position = True
                        except Exception as e:
                            print(f"Error checking position for {drone.drone_id}: {e}")
                    
                    if reached_position:
                        print(f"{drone.drone_id} reached waypoint.")
                        drone.is_moving = False
                        drone.change_led_colour((0, 0, 255))  # Blue for all drones
                        active_drones.remove(drone)
                
                if active_drones:
                    time.sleep(0.5)

    def generate_formation_positions(self, center_position, orientation, scale=1.0):
        """
        Generate pentagon formation positions for all drones based on a center position
        
        Args:
            center_position: The center point of the formation [x, y, z]
            orientation: The orientation of the formation in degrees
            scale: Scale factor to adjust formation size (1.0 = normal size)
            
        Returns:
            List of positions for each drone in the formation
        """
        center_xy = center_position[:2]  # Get only x, y coordinates
        z = center_position[2]  # Get z coordinate
        
        # Base radius for the pentagon
        if getattr(self, "using_wide_formation", True):
            # Wider formation for initial and final positioning
            base_radius = 0.8  # Reduced from 1.2 to make a smaller pentagon
        else:
            # Tighter formation while navigating
            base_radius = 0.5  # Reduced from 0.6 to make a smaller pentagon
            
        # Apply scale factor
        radius = base_radius * scale
        
        # Generate pentagon formation in 2D
        pentagon_positions_2d = []
        
        # Calculate the angle between each point in the pentagon
        angle_step = 2 * math.pi / min(self.num_drones, 5)
        
        # Generate positions for each drone
        for i in range(min(self.num_drones, 5)):
            # Calculate angle for this point, starting from the top (90 degrees)
            angle = i * angle_step - math.pi/2 + math.radians(orientation)
            
            # Calculate position
            x = center_xy[0] + radius * math.cos(angle)
            y = center_xy[1] + radius * math.sin(angle)
            
            pentagon_positions_2d.append([x, y])
        
        # Add z-coordinate to each position
        formation_positions = [pos + [z] for pos in pentagon_positions_2d]
        
        return formation_positions

    def generate_paths_for_all_drones(self, waypoints: List[List[float]]):
        """
        Generate paths for all drones based on waypoints for the formation center
        
        Args:
            waypoints: List of waypoints for the formation center
        """
        # Set paths for all drones
        all_drones = list(self.drones.values())
        
        # Clear existing paths
        for drone in all_drones:
            drone.path = []
        
        # Use the fixed formation orientation established during initial formation
        orientation_deg = getattr(self, "formation_orientation", 90.0)
        
        # For each waypoint, calculate the pentagon formation positions
        for i, wp in enumerate(waypoints):
            # Generate pentagon formation positions for this waypoint
            formation_positions = self.generate_formation_positions(wp, orientation_deg)
            
            # Assign positions to each drone's path
            for j, drone in enumerate(all_drones):
                if j < len(formation_positions):
                    drone.path.append(formation_positions[j])
                else:
                    # If we have more drones than positions, use the last position
                    drone.path.append(formation_positions[-1])
        
        # Print path information
        print(f"Generated paths for {len(all_drones)} drones with {len(waypoints)} waypoints each")
        for i, drone in enumerate(all_drones):
            print(f"Drone {i} path: {len(drone.path)} waypoints")
            
        print("Pentagon formation paths generated successfully.")

    def astar_plan_path(self, start, goal, obstacles, planning_distance=1.5, resolution=0.02, 
                       safety_margin=0.3, bounds=None):
        """A* algorithm for 3D path planning with limited planning distance
        
        Args:
            start: Start position [x, y, z]
            goal: Goal position [x, y, z]
            obstacles: List of obstacles as [x, y, z, radius]
            planning_distance: Maximum planning distance (in meters)
            resolution: Grid resolution for neighbor search
            safety_margin: Safety margin to add around obstacles (in meters)
            bounds: Environment bounds as [xmin, xmax, ymin, ymax, zmin, zmax]
            
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
        
        # If no bounds are provided, set default bounds
        if bounds is None:
            # Set bounds based on start and goal with some margin
            max_distance = max(planning_distance, math.dist(start, goal))
            bounds = [
                min(start[0], goal[0]) - max_distance,
                max(start[0], goal[0]) + max_distance,
                min(start[1], goal[1]) - max_distance,
                max(start[1], goal[1]) + max_distance,
                min(start[2], goal[2]) - max_distance,
                max(start[2], goal[2]) + max_distance
            ]
        
        # Create simpler movement directions (6-connectivity) for better performance
        movements = []
        
        # 6 basic directions
        movements.append([resolution, 0, 0])   # Right
        movements.append([-resolution, 0, 0])  # Left
        movements.append([0, resolution, 0])   # Forward
        movements.append([0, -resolution, 0])  # Backward
        movements.append([0, 0, resolution])   # Up
        movements.append([0, 0, -resolution])  # Down
        
        # Add 8 diagonal directions in the horizontal plane for better paths
        movements.append([resolution, resolution, 0])    # Right-Forward
        movements.append([resolution, -resolution, 0])   # Right-Backward
        movements.append([-resolution, resolution, 0])   # Left-Forward
        movements.append([-resolution, -resolution, 0])  # Left-Backward
        
        # Function to check if a position is within bounds
        def is_within_bounds(position):
            return (bounds[0] <= position[0] <= bounds[1] and
                    bounds[2] <= position[1] <= bounds[3] and
                    bounds[4] <= position[2] <= bounds[5])
        
        # Function to check if a position collides with any obstacle
        def is_collision_free(position):
            for obs in obstacles:
                ox, oy, oz, radius = obs
                # Check if position is inside obstacle sphere with safety margin
                dist = math.sqrt(
                    (position[0] - ox) ** 2 +
                    (position[1] - oy) ** 2 +
                    (position[2] - oz) ** 2
                )
                if dist <= radius + safety_margin:
                    return False
            return True
        
        # Function to calculate heuristic (Euclidean distance)
        def calculate_h(position):
            return math.sqrt(
                (position[0] - goal[0]) ** 2 +
                (position[1] - goal[1]) ** 2 +
                (position[2] - goal[2]) ** 2
            )
        
        # Check if distance from start to goal is greater than planning_distance
        distance_to_goal = math.sqrt(
            (start[0] - goal[0]) ** 2 +
            (start[1] - goal[1]) ** 2 +
            (start[2] - goal[2]) ** 2
        )
        
        # If goal is within planning distance, plan directly to it
        if distance_to_goal <= planning_distance:
            actual_goal = goal
        else:
            # Create an intermediate goal in the direction of the final goal
            # at distance planning_distance from start
            direction = [
                (goal[0] - start[0]) / distance_to_goal,
                (goal[1] - start[1]) / distance_to_goal,
                (goal[2] - start[2]) / distance_to_goal
            ]
            
            actual_goal = [
                start[0] + direction[0] * planning_distance,
                start[1] + direction[1] * planning_distance,
                start[2] + direction[2] * planning_distance
            ]
            
            print(f"Goal too far. Planning to intermediate point at {actual_goal}")
        
        # Check if we can see the goal directly (line of sight)
        def has_line_of_sight(from_pos, to_pos):
            # Number of checks based on distance
            dist = math.sqrt(
                (from_pos[0] - to_pos[0]) ** 2 +
                (from_pos[1] - to_pos[1]) ** 2 +
                (from_pos[2] - to_pos[2]) ** 2
            )
            
            num_checks = max(10, int(dist / (resolution * 2)))
            
            for i in range(1, num_checks):
                t = i / num_checks
                check_point = [
                    from_pos[0] + t * (to_pos[0] - from_pos[0]),
                    from_pos[1] + t * (to_pos[1] - from_pos[1]),
                    from_pos[2] + t * (to_pos[2] - from_pos[2])
                ]
                
                if not is_collision_free(check_point):
                    return False
            
            return True
        
        # If we have direct line of sight, return direct path
        if has_line_of_sight(start, actual_goal):
            print("Direct line of sight to goal found")
            return [start, actual_goal]
        
        # Create start and goal nodes
        start_node = Node(start)
        goal_node = Node(actual_goal)
        
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
        max_iterations = 10000  # Set a reasonable limit
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Find node with lowest f value in open set
            current_key = min(open_set, key=lambda k: open_set[k].f)
            current_node = open_set[current_key]
            
            # Check if we've reached the goal (within resolution distance)
            if (math.sqrt((current_node.position[0] - actual_goal[0]) ** 2 +
                         (current_node.position[1] - actual_goal[1]) ** 2 +
                         (current_node.position[2] - actual_goal[2]) ** 2) <= resolution * 2):
                
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
            
            # Every 50 iterations, check if we can see the goal from current node
            if iterations % 50 == 0:
                if has_line_of_sight(current_node.position, actual_goal):
                    # Create a direct path from here
                    path = []
                    while current_node:
                        path.append(current_node.position)
                        current_node = current_node.parent
                    
                    # Add goal node to path and return reversed path
                    path.append(actual_goal)
                    return path[::-1]
            
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
        
        print(f"A* search failed after {iterations} iterations")
        
        # Return at least a simple straight-line path toward goal
        mid_point = [
            start[0] + (actual_goal[0] - start[0]) * 0.5,
            start[1] + (actual_goal[1] - start[1]) * 0.5,
            start[2] + (actual_goal[2] - start[2]) * 0.5,
        ]
        
        # Try to move the mid point vertically to avoid obstacles
        for vert_offset in [0.3, 0.5, 0.8, 1.0]:
            adjusted_mid = [mid_point[0], mid_point[1], mid_point[2] + vert_offset]
            if is_collision_free(adjusted_mid):
                mid_point = adjusted_mid
                break
        
        return [start, mid_point, actual_goal]

    # Helper method to check line of sight between two points
    def has_line_of_sight(self, from_pos, to_pos, obstacles, resolution=0.1):
        """Check if there is a clear line of sight between two positions"""
        # Calculate distance between points
        dist = math.sqrt(
            (from_pos[0] - to_pos[0]) ** 2 +
            (from_pos[1] - to_pos[1]) ** 2 +
            (from_pos[2] - to_pos[2]) ** 2
        )
        
        # Number of checks based on distance
        num_checks = max(10, int(dist / resolution))
        
        for i in range(1, num_checks):
            t = i / num_checks
            check_point = [
                from_pos[0] + t * (to_pos[0] - from_pos[0]),
                from_pos[1] + t * (to_pos[1] - from_pos[1]),
                from_pos[2] + t * (to_pos[2] - from_pos[2])
            ]
            
            # Check collision with any obstacle
            for obs in obstacles:
                ox, oy, oz, radius = obs
                # Check if position is inside obstacle sphere
                dist_to_obs = math.sqrt(
                    (check_point[0] - ox) ** 2 +
                    (check_point[1] - oy) ** 2 +
                    (check_point[2] - oz) ** 2
                )
                if dist_to_obs <= radius + 0.3:  # Using fixed safety margin
                    return False
        
        return True

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






