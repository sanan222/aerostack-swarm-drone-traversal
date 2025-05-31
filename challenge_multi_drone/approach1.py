#!/usr/bin/env python3
"""
Centralised Leader-Follower Mission for a Swarm of 5 Drones

In this implementation, drone0 acts as the leader following a circular trajectory.
The other drones (drone1 to drone4) are followers that maintain a constant offset relative to the leader.
This code uses the BehaviorHandler to call the actual behaviors (takeoff, go_to, land) that should be available
in your patched Aerostack2/Crazyflie interface.
 
Author: [Your Name]
Based on the original code by Rafael Perez-Segui, Miguel Fernandez-Cortizas
License: BSD-3-Clause
"""

import argparse
import sys
import math
import time
import random
from math import cos, sin, pi
from typing import List, Optional
import rclpy
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA

# -------------------------------------------------------------------
# Helper: Leader Path Generation (Circular Trajectory)
# -------------------------------------------------------------------
def get_leader_path(num_points: int = 8, radius: float = 3.0, altitude: float = 1.5) -> list:
    path = []
    for i in range(num_points):
        angle = 2 * pi * i / num_points
        x = radius * cos(angle)
        y = radius * sin(angle)
        z = altitude
        path.append([x, y, z])
    return path

# Add function to generate different formation patterns
def get_formation_offsets(formation_type: str, num_followers: int = 4) -> dict:
    """Generate offsets for different formation patterns."""
    offsets = {}
    
    if formation_type == "line":
        # Line formation behind leader
        for i in range(1, num_followers + 1):
            offsets[i] = [-i * 0.8, 0.0, 0.0]
    
    elif formation_type == "v_shape":
        # V-shape formation
        for i in range(1, num_followers + 1):
            if i % 2 == 1:  # odd numbers go left
                idx = (i + 1) // 2
                offsets[i] = [-idx * 0.7, idx * 0.7, 0.0]
            else:  # even numbers go right
                idx = i // 2
                offsets[i] = [-idx * 0.7, -idx * 0.7, 0.0]
    
    elif formation_type == "diamond":
        # Diamond formation
        offsets[1] = [-0.8, 0.0, 0.0]  # behind
        offsets[2] = [0.0, 0.8, 0.0]   # right
        offsets[3] = [0.0, -0.8, 0.0]  # left
        offsets[4] = [0.8, 0.0, 0.0]   # front
    
    elif formation_type == "grid":
        # Grid formation (2x2 behind leader)
        offsets[1] = [-0.8, 0.8, 0.0]
        offsets[2] = [-0.8, -0.8, 0.0]
        offsets[3] = [-1.6, 0.8, 0.0]
        offsets[4] = [-1.6, -0.8, 0.0]
    
    else:  # default to staggered
        # Staggered formation
        offsets[1] = [-0.5, 0.5, 0.0]
        offsets[2] = [-0.5, -0.5, 0.0]
        offsets[3] = [-1.0, 0.5, 0.0]
        offsets[4] = [-1.0, -0.5, 0.0]
    
    return offsets

# -------------------------------------------------------------------
# Optional Debug: Waypoint Plotter
# -------------------------------------------------------------------
class Choreographer:
    @staticmethod
    def draw_waypoints(waypoints):
        import matplotlib.pyplot as plt
        x_vals = [wp[0] for wp in waypoints]
        y_vals = [wp[1] for wp in waypoints]
        plt.plot(x_vals, y_vals, 'o-b')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Leader Waypoints")
        plt.show()

# -------------------------------------------------------------------
# Dancer: Extended Drone Interface with Actual Behavior Calls
# -------------------------------------------------------------------
class Dancer(DroneInterface):
    def __init__(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self.namespace = namespace
        self.current_behavior: Optional[BehaviorHandler] = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        self._speed = 0.5  # using _speed to avoid conflict with read-only properties.
        self.yaw_mode = YawMode.PATH_FACING
        self.yaw_angle = None
        self.frame_id = "earth"
        
        # Wait for behaviors to be available
        self._wait_for_behaviors()
        
    def _wait_for_behaviors(self):
        """Wait for behaviors to be available."""
        print(f"[INFO] {self.namespace}: Waiting for behaviors to be available...")
        max_attempts = 10
        for attempt in range(max_attempts):
            if self.check_behavior_available("takeoff") and \
               self.check_behavior_available("go_to") and \
               self.check_behavior_available("land"):
                print(f"[INFO] {self.namespace}: All behaviors are available!")
                return
            time.sleep(1.0)
        print(f"[WARN] {self.namespace}: Not all behaviors are available after {max_attempts} attempts.")
    
    def check_behavior_available(self, behavior_name):
        """Check if a behavior is available."""
        try:
            # For checking availability:
            if behavior_name == "takeoff":
                from as2_msgs.action import Takeoff
                bh = BehaviorHandler(self.namespace, behavior_name="TakeoffBehavior")
            elif behavior_name == "go_to":
                from as2_msgs.action import GoTo
                bh = BehaviorHandler(self.namespace, behavior_name="GoToBehavior")
            elif behavior_name == "land":
                from as2_msgs.action import Land
                bh = BehaviorHandler(self.namespace, behavior_name="LandBehavior")

            else:
                print(f"[ERROR] {self.namespace}: Unknown behavior {behavior_name}")
                return False
                       
            return bh.is_available()
        except Exception as e:
            print(f"[ERROR] {self.namespace}: Error checking behavior {behavior_name}: {e}")
            return False

    def change_led_colour(self, colour):
        msg = ColorRGBA()
        msg.r = colour[0] / 255.0
        msg.g = colour[1] / 255.0
        msg.b = colour[2] / 255.0
        self.led_pub.publish(msg)

    def change_leds_random_colour(self):
        self.change_led_colour([random.randint(0, 255) for _ in range(3)])

    # --- Actual Behavior Implementations ---
    def takeoff(self, altitude, speed, wait_flag):
        print(f"[INFO] {self.namespace}: Executing takeoff behavior.")
        try:
            from as2_msgs.action import Takeoff
            bh = BehaviorHandler(self.namespace, behavior_name="takeoff")
            if not bh.is_available():
                print(f"[ERROR] {self.namespace}: Takeoff behavior not available!")
                return False
            bh.call_behavior(altitude, speed, wait_flag)
            self.current_behavior = bh
            return True
        except Exception as e:
            print(f"[ERROR] {self.namespace}: Failed to execute takeoff: {e}")
            return False

    def go_to(self, x, y, z, speed, yaw_mode, yaw_angle, frame_id, wait_flag):
        print(f"[INFO] {self.namespace}: Executing go_to behavior to [{x}, {y}, {z}].")
        try:
            from as2_msgs.action import GoTo
            bh = BehaviorHandler(self.namespace, behavior_name="go_to")
            if not bh.is_available():
                print(f"[ERROR] {self.namespace}: Go_to behavior not available!")
                return False
            bh.call_behavior(x, y, z, speed, yaw_mode, yaw_angle, frame_id, wait_flag)
            self.current_behavior = bh
            return True
        except Exception as e:
            print(f"[ERROR] {self.namespace}: Failed to execute go_to: {e}")
            return False

    def land(self, speed, wait_flag):
        print(f"[INFO] {self.namespace}: Executing land behavior.")
        try:
            from as2_msgs.action import Land
            bh = BehaviorHandler(self.namespace, behavior_name="land")
            if not bh.is_available():
                print(f"[ERROR] {self.namespace}: Land behavior not available!")
                return False
            bh.call_behavior(speed, wait_flag)
            self.current_behavior = bh
            return True
        except Exception as e:
            print(f"[ERROR] {self.namespace}: Failed to execute land: {e}")
            return False

    def do_behavior(self, beh, *args) -> bool:
        if hasattr(self, beh):
            method = getattr(self, beh)
            return method(*args)
        else:
            print(f"[ERROR] {self.namespace}: Behavior '{beh}' not implemented.")
            return False

    def go_to_point(self, point: list) -> bool:
        return self.do_behavior("go_to", point[0], point[1], point[2],
                         self._speed, self.yaw_mode, self.yaw_angle, self.frame_id, False)

    def goal_reached(self) -> bool:
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE

# -------------------------------------------------------------------
# SwarmConductor: Manages the Swarm of Drones
# -------------------------------------------------------------------
class SwarmConductor:
    def __init__(self, drones_ns: List[str], verbose: bool = False, use_sim_time: bool = False):
        self.drones: dict[int, Dancer] = {}
        
        for index, ns in enumerate(drones_ns):
            print(f"Initializing drone {index}: {ns}")
            self.drones[index] = Dancer(ns, verbose, use_sim_time)
            time.sleep(3.0)
        
        time.sleep(5.0)
        print(f"SwarmConductor initialized with {len(drones_ns)} drones")

    def shutdown(self):
        for drone in self.drones.values():
            drone.shutdown()

    def wait_all(self):
        all_finished = False
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while not all_finished and (time.time() - start_time < max_wait_time):
            all_finished = True
            for idx, drone in self.drones.items():
                if not drone.goal_reached():
                    all_finished = False
                    print(f"Waiting for drone {idx} to reach goal...")
            time.sleep(0.5)
            
        if not all_finished:
            print("Warning: Not all drones reached their goals within the timeout period!")
        return all_finished

    def get_ready(self) -> bool:
        success = True
        for idx, drone in self.drones.items():
            print(f"Preparing drone {idx} for flight...")
            arm_success = drone.arm()
            offboard_success = drone.offboard()
            if not (arm_success and offboard_success):
                print(f"Failed to prepare drone {idx}!")
            success = success and arm_success and offboard_success
            time.sleep(1.0)
        return success

    def takeoff(self):
        for idx, drone in self.drones.items():
            print(f"Taking off drone {idx}...")
            success = drone.do_behavior("takeoff", 1.0, 0.7, False)
            if success:
                drone.change_led_colour((0, 255, 0))
            time.sleep(1.0)
        self.wait_all()

    def land(self):
        for idx, drone in self.drones.items():
            print(f"Landing drone {idx}...")
            drone.do_behavior("land", 0.4, False)
            time.sleep(1.0)
        self.wait_all()

    # --- Leader-Follower Algorithm ---
    def leader_follower(self, leader_path: list, formation_offsets: dict):
        leader = self.drones[0]
        print("Starting Leader-Follower mission...")
        
        for i in range(len(self.drones)):
            if i not in self.drones:
                print(f"Error: Drone {i} not found in the swarm!")
                return
        
        for wp_idx, wp in enumerate(leader_path):
            print(f"Leader moving to waypoint {wp_idx+1}/{len(leader_path)}: {wp}")
            success = leader.go_to_point(wp)
            if not success:
                print(f"Failed to send leader to waypoint {wp_idx+1}!")
                continue
                
            leader_reached = False
            timeout = 30  # seconds
            start_time = time.time()
            
            while not leader_reached and (time.time() - start_time < timeout):
                leader_reached = leader.goal_reached()
                time.sleep(0.5)
                
            if not leader_reached:
                print(f"Leader failed to reach waypoint {wp_idx+1} within timeout!")
                continue
                
            leader.change_leds_random_colour()
            print(f"Leader reached waypoint {wp_idx+1}")
            
            for i, drone in self.drones.items():
                if i == 0:
                    continue
                if i not in formation_offsets:
                    print(f"Warning: No formation offset defined for drone {i}")
                    continue
                    
                offset = formation_offsets[i]
                target = [wp[0] + offset[0], wp[1] + offset[1], wp[2] + offset[2]]
                print(f"{drone.namespace} moving to target: {target} (offset: {offset})")
                success = drone.go_to_point(target)
                if success:
                    drone.change_leds_random_colour()
                else:
                    print(f"Failed to send {drone.namespace} to target!")
            
            if self.wait_all():
                print(f"Waypoint {wp_idx+1} reached by all drones.\n")
            else:
                print(f"Not all drones reached their positions for waypoint {wp_idx+1}.\n")
        
        print("Leader-Follower mission complete.")
    
    def change_formation(self, new_formation: str, current_leader_position: list):
        print(f"Changing formation to: {new_formation}")
        
        formation_offsets = get_formation_offsets(new_formation)
        
        for i, drone in self.drones.items():
            if i == 0:
                continue
            
            if i not in formation_offsets:
                print(f"Warning: No formation offset defined for drone {i}")
                continue
                
            offset = formation_offsets[i]
            target = [
                current_leader_position[0] + offset[0],
                current_leader_position[1] + offset[1], 
                current_leader_position[2] + offset[2]
            ]
            print(f"{drone.namespace} moving to new formation position: {target}")
            drone.go_to_point(target)
        
        self.wait_all()
        print(f"Formation changed to {new_formation}")
        return formation_offsets

# -------------------------------------------------------------------
# Utility: Confirmation Prompt
# -------------------------------------------------------------------
def confirm(msg: str = 'Continue') -> bool:
    confirmation = input(f"{msg}? (y/n): ")
    return confirmation.strip().lower() == "y"

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Leader-Follower Mission for a Drone Swarm')
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='IDs of the drones to be used in the mission')
    parser.add_argument('-w', '--world', type=str, default='world_swarm.yaml',
                        help='World to be used in the mission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-f', '--formation', type=str, default='staggered',
                        choices=['line', 'v_shape', 'diamond', 'grid', 'staggered'],
                        help='Initial formation pattern')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Enable debug mode with additional output')
    args = parser.parse_args()

    rclpy.init()
    
    if len(args.namespaces) < 3:
        print("Error: At least 3 drones are required for the mission")
        sys.exit(1)
    
    print(f"Initializing swarm with drones: {args.namespaces}")
    
    try:
        swarm = SwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)
        
        leader_path = get_leader_path(num_points=8, radius=3.0, altitude=1.5)
        
        if args.debug:
            Choreographer.draw_waypoints(leader_path)
        
        formation_offsets = get_formation_offsets(args.formation)
        
        if confirm("Prepare drones for takeoff"):
            if swarm.get_ready():
                print("All drones are armed and in offboard mode")
                
                if confirm("Takeoff"):
                    swarm.takeoff()
                    print("All drones have taken off successfully")
                    
                    if confirm("Start Leader-Follower Mission"):
                        print("\n=== Stage 1: Formation Flight with Changing Formations ===")
                        
                        half_path = leader_path[:len(leader_path)//2]
                        swarm.leader_follower(half_path, formation_offsets)
                        
                        new_formation = "diamond" if args.formation != "diamond" else "v_shape"
                        current_leader_pos = half_path[-1]
                        formation_offsets = swarm.change_formation(new_formation, current_leader_pos)
                        
                        remaining_path = leader_path[len(leader_path)//2:]
                        swarm.leader_follower(remaining_path, formation_offsets)
                        
                        print("Formation flight mission completed successfully!")
                    
                    if confirm("Land"):
                        swarm.land()
                        print("All drones have landed successfully")
            else:
                print("Error during arming/offboard setup.")
                sys.exit(1)

        print("Shutting down swarm...")
    except Exception as e:
        print(f"Error during mission execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'swarm' in locals():
            swarm.shutdown()
        rclpy.shutdown()
        sys.exit(0)

if __name__ == '__main__':
    main()
