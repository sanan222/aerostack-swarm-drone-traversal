# COMP0240 Multi-Drone Challenge CW2

## Running the codes
This file is for running Multi-Agent Path Finding algorithms in each scenario.


### For scenario 1:

#### In Terminal 1:

```bash
cd challenge_multi_drone
colcon build
source install/setup.bash
cd src/challenge_multi_drone/
./launch_as2.bash -m -s scenarios/scenario1_stage1.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

#### In Terminal 2:
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage1.yaml
```

#### In Terminal 3:

For centralized Multi Agent Path Finding (MAPF) method

```bash
python3 mapf_scenario_1.py --scenario_file scenarios/scenario1_stage1.yaml
```



### For scenario 2:
#### In Terminal 1:

```bash
./launch_as2.bash -m -s scenarios/scenario1_stage2.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

#### In Terminal 2:
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage2.yaml
```

#### In Terminal 3:

For centralized Multi Agent Path Finding (MAPF) method

```bash
python3 mapf_scenario_2.py --scenario_file scenarios/scenario1_stage2.yaml
```



### For scenario 3:
#### In Terminal 1:

```bash
./launch_as2.bash -m -s scenarios/scenario1_stage3.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

#### In Terminal 2:
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage3.yaml
```

#### In Terminal 3:

For centralized Multi Agent Path Finding (MAPF) method

```bash
python3 mapf_scenario_3.py --scenario_file scenarios/scenario1_stage3.yaml
```




### For scenario 4:
#### In Terminal 1:

```bash
./launch_as2.bash -m -s scenarios/scenario1_stage4.yaml -w config_sim/config_multicopter/world_swarm.yaml
```

#### In Terminal 2:
```bash
./launch_ground_station.bash -v -s scenarios/scenario1_stage4.yaml
```

#### In Terminal 3:

For centralized Multi Agent Path Finding (MAPF) method

```bash
python3 mapf_scenario_4.py --scenario_file scenarios/scenario1_stage4.yaml
```


