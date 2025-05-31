#!/bin/bash
set +e 

usage() {
    echo "  options:"
    echo "      -s: scenario file to load from. Default is 'scenarios/scenario1.yaml'"
    echo "      -w: world config file to use as base template. Default is 'config_sim/config/world.yaml'"
    echo "      -n: select drones namespace to launch, values are comma separated. By default, it will get all drones from world description file"
    echo "      -c: if set, the real crazyflie interface will be launched instead of the simulation. Defaults to false"
    echo "      -m: if set, it will use the multicopter simulation platform instead of gazebo"
    echo "      -g: launch using gnome-terminal instead of tmux. Default not set"
    echo "------------------"
    echo "  Note: you can set and export the environment variable CW2_SCENARIO_FILE in your ~/.bashrc to set a default variable "
}

# Initialize variables with default values
scenario_file="${CW2_SCENARIO_FILE:=scenarios/scenario1.yaml}" # Set using environment variable?? 
drones_namespace_comma=""
launch_simulation="true"
use_multicopter="false"
use_gnome="false"

# Arg parser
while getopts "s:w:n:cmg" opt; do
  case ${opt} in
    s )
      scenario_file="${OPTARG}"
      ;;
    w )
      world_config="${OPTARG}"
      ;;
    n )
      drones_namespace_comma="${OPTARG}"
      ;;
    c )
      launch_simulation="false"
      ;;
    m )
      use_multicopter="true"
      ;;
    g )
      use_gnome="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CONFIG_SIM="${SCRIPT_DIR}/config_sim"
CONFIG_REAL="${SCRIPT_DIR}/config_real"

simulation_config_folder="${CONFIG_SIM}/world"
simulation_file_name="world.yaml"
simulation_config_file="${simulation_config_folder}/${simulation_file_name}"

config_folder=""
if [[ ${launch_simulation} == "true" ]]; then
  config_folder="${CONFIG_SIM}"
  if [[ ${use_multicopter} == "true" ]]; then
    echo "Setting up for multicopter"

    # Set the world configuration
    if [ -z "$world_config" ]; then
      world_config="${CW2_WORLD_FILE:=$config_folder/config_multicopter/world.yaml}"
    fi
    # TODO
    python3 "${SCRIPT_DIR}/utils/generate_world_from_scenario.py" "${scenario_file}" -w "${config_folder}/config/world.yaml" -o "${simulation_config_folder}" -f "${simulation_file_name}"
    cp "${world_config}" "${simulation_config_file}" # Generate Models but Overwrite world. 
    drone_config="${config_folder}/config_multicopter/config.yaml"
    config_dir="${config_folder}/config_multicopter"
  else # do Gazebo
    echo "Setting up for Gazebo"

    # Ensure this folders gazebo packages are on the path for both aerostack2 and gazebo to read...
    export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:"${config_folder}/gazebo/models":"${config_folder}/gazebo/worlds":"${config_folder}/gazebo/plugins":"${simulation_config_folder}/models"
    export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:"${config_folder}/gazebo/models":"${config_folder}/gazebo/worlds":"${config_folder}/gazebo/plugins":"${simulation_config_folder}/models"
    export AS2_EXTRA_DRONE_MODELS=crazyflie_led_ring

    # Set the world configuration
    if [ -z "$world_config" ]; then
      world_config="${CW2_WORLD_FILE:=$config_folder/config/world.yaml}"
    fi

    # Generate Simulated World from configuration
    python3 "${SCRIPT_DIR}/utils/generate_world_from_scenario.py" "${scenario_file}" -w "${world_config}" -o "${simulation_config_folder}" -f "${simulation_file_name}"
    drone_config="${config_folder}/config/config.yaml"
    config_dir="${config_folder}/config"
  fi
else
  config_folder="${CONFIG_REAL}"
  drone_config="${config_folder}/config/config.yaml"
  config_dir="${config_folder}/config"
fi

# If no drone namespaces are provided, get them from the world description config file 
if [ -z "$drones_namespace_comma" ]; then

  if [[ ${launch_simulation} == "true" ]]; then
    dnamespace_lookup_file="${world_config}"
  else
    dnamespace_lookup_file="${drone_config}"
  fi

  drones_namespace_comma=$(python3 utils/get_drones.py -p ${dnamespace_lookup_file} --sep ',')
fi
IFS=',' read -r -a drone_namespaces <<< "$drones_namespace_comma"

# Select between tmux and gnome-terminal
tmuxinator_mode="start"
tmuxinator_end="wait"
tmp_file="/tmp/as2_project_launch_${drone_namespaces[@]}.txt"
if [[ ${use_gnome} == "true" ]]; then
  tmuxinator_mode="debug"
  tmuxinator_end="> ${tmp_file} && python3 utils/tmuxinator_to_genome.py -p ${tmp_file} && wait"
fi

# Launch aerostack2 for each drone namespace
for namespace in ${drone_namespaces[@]}; do
  base_launch="false"
  if [[ ${namespace} == ${drone_namespaces[0]} ]]; then
    base_launch="true"
  fi
  eval "tmuxinator ${tmuxinator_mode} -n ${namespace} -p ${SCRIPT_DIR}/tmuxinator/aerostack2.yaml \
    drone_namespace=${namespace} \
    script_folder=${SCRIPT_DIR} \
    simulation=${launch_simulation} \
    config_dir=${config_dir} \
    config_file=${drone_config} \
    simulation_config_file=${simulation_config_file} \
    base_launch=${base_launch} \
    use_multicopter=${use_multicopter} \
    multicopter_uav_config="${CONFIG_SIM}/config_multicopter/uav_config.yaml" \
    scenario_file=${scenario_file} \
    ${tmuxinator_end}"

  sleep 0.2 # Wait for tmuxinator to finish
done

# Attach to tmux session
if [[ ${use_gnome} == "false" ]]; then
  tmux attach-session -t ${drone_namespaces[0]}
# If tmp_file exists, remove it
elif [[ -f ${tmp_file} ]]; then
  rm "${tmp_file}"
fi
