#!/usr/bin/env python3

import argparse
import sys
import time
import random
import threading
from typing import List
from math import radians, cos, sin, atan2

import rclpy
from rclpy.executors import MultiThreadedExecutor
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA
# Odometry kullanılmayacağından yorum satırında.
# from nav_msgs.msg import Odometry

import yaml  # Senaryo dosyası okumak için


class Choreographer:
    """Merkezi Formasyon Yol Üreteci"""

    @staticmethod
    def delta_formation(base: float, height: float, orientation: float = 0.0,
                          center: list = [0.0, 0.0], num_drones: int = 5):
        theta = radians(orientation)
        positions = []
        if num_drones == 3:
            v0 = [-height * cos(theta)/2 - base * sin(theta)/2 + center[0],
                  base * cos(theta)/2 - height * sin(theta)/2 + center[1]]
            v1 = [height * cos(theta)/2 + center[0],
                  height * sin(theta)/2 + center[1]]
            v2 = [-height * cos(theta)/2 + base * sin(theta)/2 + center[0],
                  -base * cos(theta)/2 - height * sin(theta)/2 + center[1]]
            positions = [v0, v1, v2]
        elif num_drones == 5:
            v0 = [-height * cos(theta)/2 - base * sin(theta)/2 + center[0],
                  base * cos(theta)/2 - height * sin(theta)/2 + center[1]]
            v1 = [height * cos(theta)/2 + center[0],
                  height * sin(theta)/2 + center[1]]
            v2 = [-height * cos(theta)/2 + base * sin(theta)/2 + center[0],
                  -base * cos(theta)/2 - height * sin(theta)/2 + center[1]]
            v3 = [center[0], center[1]]
            v4 = [0, -height * sin(theta)/4 + center[1]]
            positions = [v0, v1, v2, v3, v4]
        return positions

    @staticmethod
    def v_formation(length: float, angle: float = 30.0, orientation: float = 0.0,
                    center: list = [0.0, 0.0], num_drones: int = 5):
        theta = radians(orientation)
        half_angle = radians(angle/2)
        positions = []
        if num_drones == 3:
            leader = [center[0], center[1]]
            left = [length*cos(theta+half_angle) + center[0],
                    length*sin(theta+half_angle) + center[1]]
            right = [length*cos(theta-half_angle) + center[0],
                     length*sin(theta-half_angle) + center[1]]
            positions = [leader, left, right]
        elif num_drones == 5:
            leader = [center[0], center[1]]
            left1 = [length/2*cos(theta+half_angle) + center[0],
                     length/2*sin(theta+half_angle) + center[1]]
            left2 = [length*cos(theta+half_angle) + center[0],
                     length*sin(theta+half_angle) + center[1]]
            right1 = [length/2*cos(theta-half_angle) + center[0],
                      length/2*sin(theta-half_angle) + center[1]]
            right2 = [length*cos(theta-half_angle) + center[0],
                      length*sin(theta-half_angle) + center[1]]
            positions = [leader, left1, left2, right1, right2]
        return positions

    @staticmethod
    def line_formation(length: float, orientation: float = 0.0,
                       center: list = [0.0, 0.0], num_drones: int = 5):
        theta = radians(orientation)
        positions = []
        # Drone'ların eşit aralıkta dizilmesi için spacing hesaplanır.
        spacing = length / (num_drones - 1)
        for i in range(num_drones):
            x = (i - (num_drones - 1)/2) * spacing * cos(theta) + center[0]
            y = (i - (num_drones - 1)/2) * spacing * sin(theta) + center[1]
            positions.append([x, y])
        return positions

    @staticmethod
    def diamond_formation(size: float, orientation: float = 0.0,
                          center: list = [0.0, 0.0], num_drones: int = 5):
        theta = radians(orientation)
        positions = []
        if num_drones == 5:
            front = [size*cos(theta) + center[0],
                     size*sin(theta) + center[1]]
            back = [-size*cos(theta) + center[0],
                    -size*sin(theta) + center[1]]
            left = [size*cos(theta+radians(90)) + center[0],
                    size*sin(theta+radians(90)) + center[1]]
            right = [size*cos(theta-radians(90)) + center[0],
                     size*sin(theta-radians(90)) + center[1]]
            middle = [center[0], center[1]]
            positions = [front, back, left, right, middle]
        elif num_drones == 3:
            front = [size*cos(theta) + center[0],
                     size*sin(theta) + center[1]]
            left = [size/2*cos(theta+radians(120)) + center[0],
                    size/2*sin(theta+radians(120)) + center[1]]
            right = [size/2*cos(theta-radians(120)) + center[0],
                     size/2*sin(theta-radians(120)) + center[1]]
            positions = [front, left, right]
        return positions

    @staticmethod
    def single_file_formation(spacing: float, orientation: float = 0.0,
                              leader_pos: list = [0.0, 0.0], num_drones: int = 5):
        theta = radians(orientation)
        positions = []
        # Lider pozisyonu
        positions.append([leader_pos[0], leader_pos[1]])
        for i in range(1, num_drones):
            offset = i * spacing
            x = leader_pos[0] - offset * cos(theta)
            y = leader_pos[1] - offset * sin(theta)
            positions.append([x, y])
        return positions

    @staticmethod
    def generate_circular_path(center: list, radius: float,
                               num_points: int, height: float):
        waypoints = []
        for i in range(num_points):
            angle = 2*3.14159*i/num_points
            x = center[0] + radius*cos(angle)
            y = center[1] + radius*sin(angle)
            waypoints.append([x, y, height])
        return waypoints

    @staticmethod
    def generate_window_traversal_path(start: list, window: list,
                                       exit: list, num_points: int,
                                       height: float):
        path = []
        for i in range(num_points//2):
            t = i/(num_points//2)
            x = start[0] + t*(window[0]-start[0])
            y = start[1] + t*(window[1]-start[1])
            path.append([x, y, height])
        for i in range(num_points//2, num_points):
            t = (i-num_points//2)/(num_points//2)
            x = window[0] + t*(exit[0]-window[0])
            y = window[1] + t*(exit[1]-window[1])
            path.append([x, y, height])
        return path


class SimpleDrone(DroneInterface):
    """
    Drone hareketini DroneInterface üzerinden kontrol eden sınıf.
    """
    def __init__(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self.__path = []
        self.__current = 0
        self.__frame_id = "earth"

        # LED kontrolü için yayıncı
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)

        self.formation_index = 0
        self.current_target = None
        self.is_moving = False
        self.current_behavior = None

        # Odometry kullanılmayacağından başlangıç pozisyonu sabit.
        self.current_pose = [0.0, 0.0, 0.0]

        print(f"Initialized {self.drone_id} drone")

    def set_path(self, path: list) -> None:
        self.__path = path
        self.__current = 0
        print(f"Path set for {self.drone_id}: {path}")

    def set_formation_index(self, index: int) -> None:
        self.formation_index = index

    def change_led_colour(self, colour):
        msg = ColorRGBA()
        msg.r = colour[0] / 255.0
        msg.g = colour[1] / 255.0
        msg.b = colour[2] / 255.0
        self.led_pub.publish(msg)

    def change_leds_random_colour(self):
        self.change_led_colour([random.randint(0,255) for _ in range(3)])

    def reset(self) -> None:
        self.__current = 0
        self.is_moving = False
        self.current_behavior = None

    def get_next_waypoint(self):
        if self.__current < len(self.__path):
            return self.__path[self.__current]
        return None

    def print_status(self):
        next_wp = self.get_next_waypoint()
        print(f"{self.drone_id}: Current pos: {self.current_pose}, Next wp: {next_wp}")

    def go_to_next(self) -> None:
        if self.__current < len(self.__path):
            point = self.__path[self.__current]
            print(f"{self.drone_id} current pos: {self.current_pose}, going to wp: {point}")
            speed = 0.5
            yaw_mode = YawMode.PATH_FACING
            yaw_angle = None
            frame_id = self.__frame_id
            wait_for_ready = False
            try:
                result = self.go_to(point[0], point[1], point[2],
                                    speed, yaw_mode, yaw_angle,
                                    frame_id, wait_for_ready)
                if isinstance(result, BehaviorHandler):
                    self.current_behavior = result
                else:
                    self.current_behavior = None
                    self.is_moving = True
            except Exception as e:
                print(f"Error in go_to for {self.drone_id}: {e}")
                self.current_behavior = None
                self.is_moving = True
            self.current_target = point
            self.__current += 1
            self.change_leds_random_colour()
        else:
            print(f"Warning: {self.drone_id} has no more waypoints")

    def goal_reached(self) -> bool:
        if self.current_behavior is not None:
            try:
                if self.current_behavior.status in [BehaviorStatus.IDLE, BehaviorStatus.COMPLETED]:
                    if self.current_target is not None:
                        self.current_pose = self.current_target.copy()
                    self.is_moving = False
                    return True
                return False
            except AttributeError:
                self.current_behavior = None
        if self.current_target is not None:
            self.current_pose = self.current_target.copy()
        self.is_moving = False
        return True

    def has_more_waypoints(self) -> bool:
        return self.__current < len(self.__path)

    def takeoff(self, height: float = 1.0):
        print(f"{self.drone_id} taking off to {height}")
        try:
            result = super().takeoff(height, 0.5, False)
            if isinstance(result, BehaviorHandler):
                self.current_behavior = result
            else:
                self.current_behavior = None
            self.is_moving = True
        except Exception as e:
            print(f"Error in takeoff for {self.drone_id}: {e}")
            self.current_behavior = None
        self.change_led_colour((0,255,0))

    def land(self):
        print(f"{self.drone_id} landing")
        try:
            result = super().land(0.5, False)
            if isinstance(result, BehaviorHandler):
                self.current_behavior = result
            else:
                self.current_behavior = None
            self.is_moving = True
        except Exception as e:
            print(f"Error in land for {self.drone_id}: {e}")
            self.current_behavior = None
            self.is_moving = True


class SwarmConductor:
    """Lider yol üretimi ile merkezi swarm kontrolü"""
    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False, scenario_file: str = None):
        self.num_drones = len(drones_ns)
        self.drones: dict[int, SimpleDrone] = {}
        for index, name in enumerate(drones_ns):
            self.drones[index] = SimpleDrone(name, verbose, use_sim_time)
            self.drones[index].set_formation_index(index)
        self.leader_index = 0
        self.current_formation = "v"
        self.formation_size = 1.5
        self.formation_height = 1.0
        self.scenario_stage = 1
        self.scenario_data = None
        if scenario_file:
            try:
                with open(scenario_file, "r") as f:
                    self.scenario_data = yaml.safe_load(f)
                print(f"Loaded scenario file: {scenario_file}")
            except Exception as e:
                print(f"Failed to load scenario file: {e}")
        print(f"SwarmConductor initialized with {self.num_drones} drones")

    def shutdown(self):
        for drone in self.drones.values():
            drone.shutdown()

    def reset_point(self):
        for drone in self.drones.values():
            drone.reset()

    def wait(self):
        print("Waiting for all drones to reach their goals...")
        start_time = time.time()
        timeout = 20
        while time.time()-start_time < timeout:
            all_reached = True
            for drone in self.drones.values():
                if drone.is_moving and not drone.goal_reached():
                    all_reached = False
                    break
            if all_reached:
                print("All drones reached their goals")
                return True
            time.sleep(0.5)
        print("Warning: Wait timeout reached! Continuing with next waypoint...")
        for drone in self.drones.values():
            if drone.is_moving:
                drone.is_moving = False
        return False

    def get_ready(self) -> bool:
        success = True
        for idx, drone in self.drones.items():
            print(f"Getting drone {idx} ready...")
            try:
                success_arm = drone.arm()
                print(f"Drone {idx} arm result: {success_arm}")
            except Exception as e:
                print(f"Error arming drone {idx}: {e}")
                if hasattr(drone, 'send_command'):
                    try:
                        drone.send_command("arm", True)
                        success_arm = True
                        print(f"Drone {idx} armed via command")
                    except Exception as e2:
                        print(f"Alternative arming failed: {e2}")
                        success_arm = False
                else:
                    success_arm = True
            try:
                success_offboard = drone.offboard()
                print(f"Drone {idx} offboard result: {success_offboard}")
            except Exception as e:
                print(f"Error setting offboard mode for drone {idx}: {e}")
                if hasattr(drone, 'send_command'):
                    try:
                        drone.send_command("offboard", True)
                        success_offboard = True
                        print(f"Drone {idx} set to offboard via command")
                    except Exception as e2:
                        print(f"Alternative offboard setting failed: {e2}")
                        success_offboard = False
                else:
                    success_offboard = True
            success = success and success_arm and success_offboard
        return success

    def takeoff(self):
        print("Taking off all drones...")
        for idx, drone in self.drones.items():
            print(f"Taking off drone {idx}...")
            drone.takeoff(1.0)
            drone.change_led_colour((0,255,0))
        self.wait()
        print("All drones have taken off")

    def land(self):
        print("Landing all drones...")
        for drone in self.drones.values():
            drone.land()
        self.wait()
        print("All drones have landed")

    def change_formation(self, formation_type: str):
        self.current_formation = formation_type
        print(f"Changed formation to: {formation_type}")

    def generate_formation_positions(self, leader_pos: list, orientation: float = 0.0):
        center = leader_pos[:2] if len(leader_pos)>2 else leader_pos
        if self.current_formation=="v":
            positions = Choreographer.v_formation(self.formation_size, 30.0, orientation, center, self.num_drones)
        elif self.current_formation=="line":
            positions = Choreographer.line_formation(self.formation_size*2, orientation, center, self.num_drones)
        elif self.current_formation=="delta":
            positions = Choreographer.delta_formation(self.formation_size, self.formation_size, orientation, center, self.num_drones)
        elif self.current_formation=="diamond":
            positions = Choreographer.diamond_formation(self.formation_size, orientation, center, self.num_drones)
        elif self.current_formation=="single_file":
            positions = Choreographer.single_file_formation(self.formation_size, orientation, leader_pos=leader_pos[:2], num_drones=self.num_drones)
        height = leader_pos[2] if len(leader_pos)>2 else self.formation_height
        return [pos+[height] for pos in positions]

    def generate_simple_path(self):
        height = 1.5
        return [
            [0.0, 0.0, height],
            [2.0, 0.0, height],
            [2.0, 2.0, height],
            [0.0, 2.0, height],
            [0.0, 0.0, height]
        ]

    def generate_leader_path_stage1(self):
        center = [0.0,0.0]
        radius = 3.0
        num_points = 16
        height = 1.5
        return Choreographer.generate_circular_path(center, radius, num_points, height)

    def interpolate_segment(self, pt1: list, pt2: list, n: int) -> List[list]:
        seg = []
        for i in range(n):
            t = i/(n-1)
            pt = [pt1[j] + t*(pt2[j]-pt1[j]) for j in range(3)]
            seg.append(pt)
        return seg

    def generate_leader_path_stage2(self):
        if self.scenario_data is not None:
            try:
                stage2 = self.scenario_data["stage2"]
                stage_center = stage2["stage_center"]  # Örneğin [0.0, -6.0]
                windows = stage2["windows"]
                sorted_keys = sorted(windows.keys(), key=lambda k: int(k))
                window_list = []
                for key in sorted_keys:
                    w = windows[key]
                    swapped = [w["center"][1], w["center"][0]]  # [x, y]
                    global_center = [stage_center[0]+swapped[0], stage_center[1]+swapped[1]]
                    height = w["height"]
                    window_list.append([global_center[0], global_center[1], height])
                offset = 3.0
                first = window_list[0]
                start = [first[0]-offset, first[1], first[2]]
                last = window_list[-1]
                exit_ = [last[0]+offset, last[1], last[2]]
                points_per_segment = 10
                path = []
                seg = self.interpolate_segment(start, first, points_per_segment)
                path.extend(seg[:-1])
                for i in range(len(window_list)-1):
                    seg = self.interpolate_segment(window_list[i], window_list[i+1], points_per_segment)
                    path.extend(seg[:-1])
                seg = self.interpolate_segment(last, exit_, points_per_segment)
                path.extend(seg)
                print(f"Stage 2 path: start={start}, windows={window_list}, exit={exit_}")
                return path
            except Exception as e:
                print(f"Error generating stage 2 path from YAML: {e}")
                import traceback
                traceback.print_exc()
        print("Using fallback path for stage 2")
        return [[-3.0,0.0,2.0],[1.5,-0.5,2.0],[3.0,0.0,2.0]]

    def generate_leader_path_stage3(self):
        height = 1.5
        return [
            [-3.0,-3.0,height],
            [-1.5,-1.5,height],
            [0.0,0.0,height],
            [1.5,1.5,height],
            [3.0,3.0,height]
        ]

    def generate_leader_path_stage4(self):
        height = 1.5
        return [
            [-3.0,-3.0,height],
            [-2.0,-2.0,height],
            [-1.0,-1.0,height],
            [0.0,0.0,height],
            [1.0,1.0,height],
            [2.0,2.0,height],
            [3.0,3.0,height]
        ]

    def generate_paths_for_all_drones(self):
        print(f"Generating paths for scenario stage {self.scenario_stage}")
        if self.scenario_stage==1:
            leader_path = self.generate_leader_path_stage1()
            formation_changes = {0:"v",4:"line",8:"delta",12:"diamond"}
        elif self.scenario_stage==2:
            leader_path = self.generate_leader_path_stage2()
            # Senaryo 2 için formationı "line" olarak belirleyip formation_size'ı da küçültüyoruz.
            self.current_formation = "line"
            self.formation_size = 0.5  # Bu değer window gap'ına uygun ayarlanabilir.
            formation_changes = {}
        elif self.scenario_stage==3:
            leader_path = self.generate_leader_path_stage3()
            formation_changes = {0:"v",2:"diamond",4:"v"}
        elif self.scenario_stage==4:
            leader_path = self.generate_leader_path_stage4()
            formation_changes = {0:"v",3:"diamond",6:"v"}
        else:
            leader_path = self.generate_simple_path()
            formation_changes = {0:"v"}
        print(f"Leader path: {leader_path}")
        self.drones[self.leader_index].set_path(leader_path)
        follower_paths = {i: [] for i in range(self.num_drones) if i != self.leader_index}
        for i in range(len(leader_path)):
            if i in formation_changes:
                self.current_formation = formation_changes[i]
                print(f"Changing formation to {self.current_formation} at wp {i}")
            orientation = 0.0
            if i < len(leader_path)-1:
                dx = leader_path[i+1][0]-leader_path[i][0]
                dy = leader_path[i+1][1]-leader_path[i][1]
                orientation = atan2(dy,dx)*180/3.14159
            formation_positions = self.generate_formation_positions(leader_path[i], orientation)
            for drone_idx in follower_paths.keys():
                if self.current_formation=="single_file":
                    formation_idx = drone_idx
                else:
                    formation_idx = drone_idx if drone_idx < self.leader_index else drone_idx-1
                if formation_idx < len(formation_positions):
                    follower_paths[drone_idx].append(formation_positions[formation_idx])
        for drone_idx, path in follower_paths.items():
            print(f"Setting path for drone {drone_idx}: {path}")
            self.drones[drone_idx].set_path(path)

    def set_scenario(self, stage: int):
        self.scenario_stage = stage
        print(f"Setting scenario to stage {stage}")

    def dance(self):
        print("Starting dance routine...")
        self.reset_point()
        self.generate_paths_for_all_drones()
        leader = self.drones[self.leader_index]
        while any(drone.has_more_waypoints() for drone in self.drones.values()):
            print("\n--- Drone statuses before next wp ---")
            for idx, drone in self.drones.items():
                drone.print_status()
            print("\nMoving to next waypoints...")
            if leader.has_more_waypoints():
                leader.go_to_next()
            for idx, drone in self.drones.items():
                if idx != self.leader_index and drone.has_more_waypoints():
                    drone.go_to_next()
            reached = self.wait()
            if not reached:
                print("Proceeding to next wp...")
            else:
                time.sleep(0.5)
        print("Dance routine completed")


def confirm(msg: str = 'Continue') -> bool:
    return input(f"{msg}? (y/n): ").lower() in ["y","yes"]


def main():
    parser = argparse.ArgumentParser(description='Multi-drone formation flight mission')
    parser.add_argument('-n','--namespaces', nargs='+',
                        default=['drone0','drone1','drone2','drone3','drone4'],
                        help='Namespaces of the drones')
    parser.add_argument('-w','--world', type=str, default='world_swarm.yaml',
                        help='World file to be used')
    parser.add_argument('-v','--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s','--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-t','--scenario', type=int, default=1,
                        help='Scenario stage (1-4)')
    parser.add_argument('--scenario_file', type=str, default='scenario1_stage2.yaml',
                        help='Path to scenario YAML file (for stage 2)')
    args = parser.parse_args()
    drones_namespace = args.namespaces
    rclpy.init()
    swarm = SwarmConductor(drones_namespace, verbose=args.verbose, use_sim_time=args.use_sim_time, scenario_file=args.scenario_file)
    swarm.set_scenario(args.scenario)
    executor = MultiThreadedExecutor()
    for drone_obj in swarm.drones.values():
        executor.add_node(drone_obj)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    if confirm("Takeoff"):
        success = swarm.get_ready()
        print(f"Get ready result: {success}")
        swarm.takeoff()
        if confirm("Start mission"):
            swarm.dance()
            while confirm("Replay"):
                swarm.dance()
        confirm("Land")
        swarm.land()
    print("Shutdown")
    swarm.shutdown()
    executor.shutdown()
    spin_thread.join()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()


