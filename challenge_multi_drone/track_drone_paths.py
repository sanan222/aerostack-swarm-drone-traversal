#!/usr/bin/env python3

import argparse
import sys
import time
import threading
import math
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import rclpy
from rclpy.executors import MultiThreadedExecutor
from as2_python_api.drone_interface import DroneInterface


class DronePathTracker:
    def __init__(self, drone_namespaces: List[str], scenario_file: str = None, 
                 sampling_interval: float = 1.0, use_sim_time: bool = False):
        """
        Initialize the drone path tracker.
        
        Args:
            drone_namespaces: List of drone namespace strings
            scenario_file: Path to the scenario file with window definitions
            sampling_interval: Time in seconds between position samples
            use_sim_time: Whether to use simulation time
        """
        print(f"Initializing DronePathTracker with {len(drone_namespaces)} drones")
        
        # Initialize ROS 2
        rclpy.init(args=None)
        
        # Initialize executor
        self.executor = MultiThreadedExecutor()
        
        # Create drone interface objects
        self.drones = {}
        for idx, ns in enumerate(drone_namespaces):
            self.drones[idx] = DroneInterface(ns, use_sim_time=use_sim_time)
            self.executor.add_node(self.drones[idx])
        
        # Add executor thread
        self.executor_thread = threading.Thread(target=self._spin_executor)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        # Trajectory tracking parameters
        self.tracking_active = False
        self.tracking_interval = sampling_interval
        self.trajectory_data = {idx: [] for idx in range(len(drone_namespaces))}
        
        # Window information
        self.windows = {}
        self.stage_center = (0, 0)
        
        # Load scenario data if provided
        if scenario_file:
            self._load_scenario_data(scenario_file)
    
    def _load_scenario_data(self, scenario_file):
        """Load window information from the scenario file."""
        try:
            print(f"Loading scenario file: {scenario_file}")
            with open(scenario_file, "r") as f:
                scenario_data = yaml.safe_load(f)
            
            if "stage2" in scenario_data:
                stage2 = scenario_data["stage2"]
                self.stage_center = stage2.get("stage_center", (0, 0))
                
                if "windows" in stage2:
                    self.windows = stage2["windows"]
                    window_ids = sorted(self.windows.keys(), key=lambda k: int(k))
                    print(f"Loaded {len(window_ids)} windows from scenario data")
                    
                    for wid in window_ids:
                        print(f"  Window {wid}: {self.windows[wid]}")
            
        except Exception as e:
            print(f"Error loading scenario file: {e}")
            import traceback
            traceback.print_exc()
    
    def _spin_executor(self):
        """Thread function to spin the ROS executor."""
        while rclpy.ok():
            self.executor.spin_once(timeout_sec=0.1)
            time.sleep(0.01)
    
    def start_tracking(self):
        """Start tracking drone positions."""
        if self.tracking_active:
            print("Position tracking is already active")
            return
        
        self.tracking_active = True
        self.tracking_thread = threading.Thread(target=self._track_positions)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        print(f"Started position tracking (sampling every {self.tracking_interval}s)")
    
    def stop_tracking(self):
        """Stop tracking drone positions."""
        if not self.tracking_active:
            print("Position tracking is not active")
            return
        
        self.tracking_active = False
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.join(timeout=2.0)
        print("Stopped position tracking")
        
        # Print statistics about collected data
        total_points = sum(len(points) for points in self.trajectory_data.values())
        print(f"Collected {total_points} total position points:")
        for idx, points in self.trajectory_data.items():
            print(f"  Drone {idx}: {len(points)} points")
    
    def _track_positions(self):
        """Thread function to record drone positions at regular intervals."""
        try:
            while self.tracking_active and rclpy.ok():
                start_time = time.time()
                
                # Record current position of each drone
                for idx, drone in self.drones.items():
                    try:
                        position = drone.position
                        self.trajectory_data[idx].append((position, time.time()))
                    except Exception as e:
                        print(f"Error recording position for drone {idx}: {e}")
                
                # Sleep to maintain the desired sampling interval
                elapsed = time.time() - start_time
                sleep_time = max(0.01, self.tracking_interval - elapsed)
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"Error in position tracking thread: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_trajectories(self, show_2d=True, show_3d=True, save_path=None):
        """
        Visualize the drone trajectories in 2D and/or 3D.
        
        Args:
            show_2d: Whether to show 2D plot (XY plane)
            show_3d: Whether to show 3D plot
            save_path: Optional path to save the visualization image
        """
        # Check if we have trajectory data
        total_points = sum(len(data) for data in self.trajectory_data.values())
        if total_points == 0:
            print("No trajectory data available for visualization")
            return
        
        print(f"Visualizing {total_points} trajectory points")
        
        # Define colors for different drones
        drone_colors = ['#00FFFF', '#FF00FF', '#0000FF', '#FF8000', '#00FF00', 
                       '#FF0000', '#00FF80', '#8000FF', '#FFFF00', '#FF0080']
        
        # Create subplots based on what's requested
        if show_2d and show_3d:
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(121)  # 2D plot
            ax2 = fig.add_subplot(122, projection='3d')  # 3D plot
        elif show_2d:
            fig = plt.figure(figsize=(12, 10))
            ax1 = fig.add_subplot(111)
            ax2 = None
        elif show_3d:
            fig = plt.figure(figsize=(12, 10))
            ax1 = None
            ax2 = fig.add_subplot(111, projection='3d')
        else:
            print("No plots requested (both show_2d and show_3d are False)")
            return
        
        # Set up 2D plot
        if show_2d:
            ax1.set_title('Drone Trajectories (Top View)', fontsize=16, pad=20)
            ax1.set_xlabel('X (m)', fontsize=12, labelpad=10)
            ax1.set_ylabel('Y (m)', fontsize=12, labelpad=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot windows in 2D
            self._plot_windows_2d(ax1)
        
        # Set up 3D plot
        if show_3d:
            ax2.set_title('Drone Trajectories (3D View)', fontsize=16, pad=20)
            ax2.set_xlabel('X (m)', fontsize=12, labelpad=10)
            ax2.set_ylabel('Y (m)', fontsize=12, labelpad=10)
            ax2.set_zlabel('Z (m)', fontsize=12, labelpad=10)
            
            # Plot windows in 3D
            self._plot_windows_3d(ax2)
        
        # Plot each drone's trajectory
        for idx, trajectory in self.trajectory_data.items():
            if not trajectory:
                continue
            
            # Extract position data, ignoring timestamps for plotting
            positions = [pos for pos, _ in trajectory]
            
            # Extract X, Y, Z coordinates
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            z_coords = [pos[2] for pos in positions]
            
            color = drone_colors[idx % len(drone_colors)]
            label = f"Drone {idx}"
            
            if show_2d:
                ax1.plot(x_coords, y_coords, '-', color=color, label=label, linewidth=2, marker='o', markersize=4)
                
                # Add drone start and end positions
                ax1.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8)
                ax1.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8)
            
            if show_3d:
                ax2.plot3D(x_coords, y_coords, z_coords, '-', color=color, label=label, linewidth=2)
                ax2.scatter3D(x_coords[0], y_coords[0], z_coords[0], color=color, s=50, marker='o')  # Start
                ax2.scatter3D(x_coords[-1], y_coords[-1], z_coords[-1], color=color, s=50, marker='s')  # End
        
        # Add legends
        if show_2d:
            ax1.legend(loc='best', fontsize=10)
        if show_3d:
            ax2.legend(loc='best', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Show plot
        plt.show()
    
    def _plot_windows_2d(self, ax):
        """Plot windows in 2D."""
        if not self.windows:
            return
        
        y0, x0 = self.stage_center  # (y, x) format in scenario file
        wall_length = 6.0  # Length of wall to draw on each side of window
        wall_thickness = 0.3  # Thickness of walls for visibility
        
        for wid, window in self.windows.items():
            wy, wx = window["center"]  # (y, x) format in scenario file
            width = window["gap_width"]
            height = window["height"]
            
            # Calculate window center in global coordinates
            center_y = y0 + wy
            center_x = x0 + wx
            half_width = width / 2
            
            # Draw the wall (horizontal orientation)
            ax.fill_between(
                [center_x - half_width - wall_length, center_x + half_width + wall_length],
                [center_y - wall_thickness], [center_y + wall_thickness],
                color='#8B4513', alpha=0.8
            )
            
            # Draw the window gap
            ax.fill_between(
                [center_x - half_width, center_x + half_width],
                [center_y - wall_thickness], [center_y + wall_thickness],
                color='white', alpha=1.0
            )
            
            # Add window number label
            ax.text(center_x, center_y + 0.5, f"Window {wid}", color='red', fontsize=10,
                   ha='center', va='center', weight='bold')
    
    def _plot_windows_3d(self, ax):
        """Plot windows in 3D."""
        if not self.windows:
            return
        
        y0, x0 = self.stage_center  # (y, x) format in scenario file
        wall_length = 6.0  # Length of wall to draw on each side of window
        
        for wid, window in self.windows.items():
            wy, wx = window["center"]  # (y, x) format in scenario file
            width = window["gap_width"]
            height = window["height"]
            distance_floor = window["distance_floor"]
            
            # Calculate window center in global coordinates
            center_y = y0 + wy
            center_x = x0 + wx
            center_z = distance_floor + height / 2
            
            half_width = width / 2
            half_height = height / 2
            
            # Draw the window frame
            # We'll create rectangles for left wall, right wall, top wall, bottom wall
            
            # Wall vertices - (x, y, z) format
            # Left wall
            left_wall = np.array([
                [center_x - half_width - wall_length, center_y, distance_floor],
                [center_x - half_width, center_y, distance_floor],
                [center_x - half_width, center_y, distance_floor + height],
                [center_x - half_width - wall_length, center_y, distance_floor + height]
            ])
            
            # Right wall
            right_wall = np.array([
                [center_x + half_width, center_y, distance_floor],
                [center_x + half_width + wall_length, center_y, distance_floor],
                [center_x + half_width + wall_length, center_y, distance_floor + height],
                [center_x + half_width, center_y, distance_floor + height]
            ])
            
            # Top wall
            top_wall = np.array([
                [center_x - half_width - wall_length, center_y, distance_floor + height],
                [center_x + half_width + wall_length, center_y, distance_floor + height],
                [center_x + half_width + wall_length, center_y, distance_floor + height + 0.5],
                [center_x - half_width - wall_length, center_y, distance_floor + height + 0.5]
            ])
            
            # Bottom wall
            bottom_wall = np.array([
                [center_x - half_width - wall_length, center_y, distance_floor],
                [center_x + half_width + wall_length, center_y, distance_floor],
                [center_x + half_width + wall_length, center_y, distance_floor - 0.5],
                [center_x - half_width - wall_length, center_y, distance_floor - 0.5]
            ])
            
            # Create 3D polygons for each wall section
            wall_color = '#8B4513'  # Brown color for walls
            alpha = 0.6  # Transparency
            
            for wall in [left_wall, right_wall, top_wall, bottom_wall]:
                poly = Poly3DCollection([wall], alpha=alpha)
                poly.set_facecolor(wall_color)
                poly.set_edgecolor('black')
                ax.add_collection3d(poly)
            
            # Add window number label in 3D
            ax.text(center_x, center_y, center_z + 1.0, f"Window {wid}", 
                   color='red', fontsize=10, ha='center', va='center', weight='bold')
    
    def shutdown(self):
        """Shutdown the tracker and clean up resources."""
        if self.tracking_active:
            self.stop_tracking()
        
        # Shutdown ROS interfaces
        for drone in self.drones.values():
            drone.shutdown()
        
        # Shutdown ROS 2
        rclpy.shutdown()
        print("DronePathTracker shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='Track and visualize drone paths during a mission')
    parser.add_argument('--drones', nargs='+', default=['drone0', 'drone1', 'drone2'],
                        help='List of drone namespaces')
    parser.add_argument('--scenario', type=str, default='src/challenge_multi_drone/config_sim/challenge.yaml',
                        help='Path to scenario file with window definitions')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds')
    parser.add_argument('--sim-time', action='store_true',
                        help='Use simulation time')
    parser.add_argument('--2d-only', dest='show_2d_only', action='store_true',
                        help='Show only 2D visualization')
    parser.add_argument('--3d-only', dest='show_3d_only', action='store_true',
                        help='Show only 3D visualization')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization image')
    
    args = parser.parse_args()
    
    try:
        # Create tracker
        tracker = DronePathTracker(
            args.drones,
            scenario_file=args.scenario,
            sampling_interval=args.interval,
            use_sim_time=args.sim_time
        )
        
        # Start tracking
        tracker.start_tracking()
        
        print("\nTracking drone positions. Press Ctrl+C to stop and show visualization...\n")
        
        # Keep running until user interrupts
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt. Stopping tracking and generating visualization...")
        
        # Stop tracking
        tracker.stop_tracking()
        
        # Show visualization based on arguments
        show_2d = not args.show_3d_only
        show_3d = not args.show_2d_only
        
        tracker.visualize_trajectories(
            show_2d=show_2d,
            show_3d=show_3d,
            save_path=args.output
        )
        
        # Shutdown
        tracker.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 