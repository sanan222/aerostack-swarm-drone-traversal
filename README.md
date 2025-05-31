# Drone Swarm Path Planning Projects with Aerostack2

## Running Instructions

This project supports centralized Multi-Agent Path Finding (MAPF) for drone swarms in simulation. Each scenario is launched and controlled using separate terminal sessions.

### Scenario 1 – Circular Path with Formation Changes

**Terminal 1:**
```bash
cd challenge_multi_drone
colcon build
source install/setup.bash
cd src/challenge_multi_drone/
./launch_as2.bash -m -s scenarios/scenario1_stage1.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

**Terminal 2:**
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage1.yaml
```

**Terminal 3:**
```bash
python3 mapf_scenario_1.py --scenario_file scenarios/scenario1_stage1.yaml
```

### Scenario 2 – Window Traversal

**Terminal 1:**
```bash
./launch_as2.bash -m -s scenarios/scenario1_stage2.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

**Terminal 2:**
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage2.yaml
```

**Terminal 3:**
```bash
python3 mapf_scenario_2.py --scenario_file scenarios/scenario1_stage2.yaml
```

### Scenario 3 – Forest Navigation

**Terminal 1:**
```bash
./launch_as2.bash -m -s scenarios/scenario1_stage3.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

**Terminal 2:**
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage3.yaml
```

**Terminal 3:**
```bash
python3 mapf_scenario_3.py --scenario_file scenarios/scenario1_stage3.yaml
```

### Scenario 4 – Dynamic Obstacle Avoidance

**Terminal 1:**
```bash
./launch_as2.bash -m -s scenarios/scenario1_stage4.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

**Terminal 2:**
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage4.yaml
```

**Terminal 3:**
```bash
python3 mapf_scenario_4.py --scenario_file scenarios/scenario1_stage4.yaml
```

---

## MAPF Methodology Overview

The Multi-Agent Path Finding (MAPF) implementation provides centralized control over drone swarms. It ensures safe, collision-free trajectories while preserving formation structure using path planning techniques such as Conflict-Based Search (CBS), Dijkstra’s algorithm, and hybrid local avoidance.

### ▸ Scenario 1: Coordinated Formation Transitions on a Circular Trajectory
- Circular path (radius = 2.0m, altitude = 1.5m) is segmented, each with a unique formation (Line, V, Grid, etc.).
- CBS computes safe, layered altitude assignments for formation transitions.
- Orientation is realigned tangentially to the path for visual consistency.
- Altitude separation and CBS ensure conflict-free motion during realignment.

### ▸ Scenario 2: Window Traversal with Adaptive Compression
- Formation dynamically compresses to fit through narrow windows using YAML-defined waypoints.
- The compression radius is calculated by:
  ```math
  r_\text{compressed} = \frac{\text{window width} - 2 \times \text{safety margin}}{2}
  ```
- Compressed formation used only at critical waypoints; standard pentagon used otherwise.
- Movement synchronization is achieved using delayed non-blocking commands for smooth transitions.

### ▸ Scenario 3: Obstacle-Aware Navigation through Forest Environments
- Environment is represented as a graph; obstacles become nodes.
- Dijkstra’s algorithm finds shortest global path; local 3D A* plans for each drone.
- Formation may split/rejoin based on obstacle proximity.
- Safety ensured using 0.2m resolution and 0.3m clearance margins.

### ▸ Scenario 4: Hybrid Navigation with Dynamic Obstacle Prediction
- Combines global A* path generation and local potential field-based avoidance.
- Real-time obstacle tracking via ROS2 topic `/dynamic_obstacles/locations`.
- Predictive "no-fly zones" built from smoothed obstacle velocity estimations.
- Movement vector is the sum of attractive (goal-seeking) and repulsive (avoidance) forces.
- Formation compresses or expands depending on proximity to moving obstacles.
- Motion is velocity-normalized and capped to 0.5 m/s for smooth, stable transitions.

---

This MAPF-based centralized control system was developed by **Sanan Garayev** and validated in simulation across all four test scenarios using five drones within Aerostack2 and ROS2 environments.
