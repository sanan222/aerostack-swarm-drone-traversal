// Copyright 2024 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


/*!*******************************************************************************************
 *  \file       crazyflie_swarm_launch.cpp
 *  \brief      Runs the crazyflie_platform swarm.
 *  \authors    Miguel Fernández Cortizas
 *
 *  \copyright  Copyright (c) 2022 Universidad Politécnica de Madrid
 *              All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************************/

#include <iostream>
#include <string>

#include <rclcpp/parameter_map.hpp>
#include <rclcpp/utilities.hpp>
#include "as2_core/core_functions.hpp"
#include "as2_core/node.hpp"
#include "as2_platform_crazyflie/crazyflie_platform.hpp"
#include "as2_core/utils/yaml_utils.hpp"

#define SWARM_ARG_NAME "swarm_config_file"
#define PARAMS_ARG_NAME "params-file"
#define URI_ARG_NAME "uri"
#define MEDIUM_FREQ_NODE 75

/****************
The structure of the params file can be in three ways:

example1.yaml

/**:
  node_name:
    ros__parameters:
      uri: <your_uri>

example2.yaml

/cf1:
  platform:
    ros__parameters:
      uri: <your_uri>


example3.yaml

/cf1:
  ros__parameters:
    uri: <your_uri>

In the first example shall raise an error, since not namespaces has been given.
In both the second and the third case, the code shall parse all the namespaces
and collect them .
 *****/

std::string find_argument_value(const std::string & argument, int argc, char ** argv)
{
  std::string res = "";
  std::string arg = "--" + argument;
  for (int i = 0; i < argc; i++) {
    if (arg == argv[i]) {
      if (i + 1 < argc) {
        res = argv[i + 1];
        return res;
      }
    }
  }
  return res;
}

YAML::Node traverse_map(const YAML::Node & node, const std::string & key)
{
  YAML::Node res;
  if (node[key]) {
    res = node[key];
  } else {
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
      if (it->second.IsMap()) {
        res = traverse_map(it->second, key);
        if (res) {
          break;
        }
      }
    }
  }
  return res;
}


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  std::vector<rclcpp::Node::SharedPtr> nodes;
  std::string swarm_config_file = find_argument_value(SWARM_ARG_NAME, argc, argv);
  if (swarm_config_file.empty()) {
    std::string params_file = find_argument_value(PARAMS_ARG_NAME, argc, argv);
    try {
      YAML::Node params = YAML::LoadFile(params_file);
      YAML::Node swarm_path = traverse_map(params, SWARM_ARG_NAME);
      if (swarm_path) {
        swarm_config_file = swarm_path.as<std::string>();
      }
    } catch (std::exception & e) {
      std::cout << "Error reading file: " << e.what() << std::endl;
      return 1;
    }
  }
  if (swarm_config_file.empty()) {
    std::cout
      << "Swarm config file not found. Please provide it as an argument or in the params file."
      << std::endl;
    return 1;
  }
  std::cout << "Swarm config file: " << swarm_config_file << std::endl;

  // find all the namespace in the config file
  YAML::Node swarm_config = YAML::LoadFile(swarm_config_file);
  for (YAML::const_iterator it = swarm_config.begin(); it != swarm_config.end(); ++it) {
    std::string name = it->first.as<std::string>();
    if (name.find("/**") != std::string::npos) {
      continue;
    }
    YAML::Node node_config = it->second;
    auto result = traverse_map(node_config, "uri").as<std::string>();
    if (result == "null") {
      continue;
    }

    // std::cout << "Drone: " << name << " at uri: " << result << " found." << std::endl;
    nodes.emplace_back(std::make_shared<CrazyfliePlatform>(name));
  }

  if (nodes.empty()) {
    std::cout << "No nodes created. Exiting." << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor;

  rclcpp::Rate r(static_cast<int>(MEDIUM_FREQ_NODE * nodes.size()));
  while (rclcpp::ok()) {
    for (auto & node : nodes) {
      executor.spin_node_some(node);
      r.sleep();
    }
  }

  rclcpp::shutdown();
  return 0;
}
