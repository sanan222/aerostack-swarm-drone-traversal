#pragma once
#ifndef GAZEBO_PLUGINS__GAZEBO_ROS_TEMPLATE_HPP_
#define GAZEBO_PLUGINS__GAZEBO_ROS_TEMPLATE_HPP_

#include <string>
#include <vector>
#include <map>

#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/World.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/SdfEntityCreator.hh>
#include <gz/math.hh>

#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/ParentLinkName.hh>
#include <gz/sim/components/ChildLinkName.hh>
#include <gz/sim/components/JointType.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Light.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Material.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Visual.hh>
#include <gz/sim/components/VisualCmd.hh>

#include <sdf/Joint.hh>
#include <sdf/Geometry.hh>
#include <sdf/Material.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Visual.hh>
#include <sdf/Light.hh>
// #include <gz/msgs/msgs.hh>
// #include <gz/sensors/Sensor.hh>

// #include <gazebo_ros/node.hpp>
// #include <rclcpp/rclcpp.hpp>
// #include <std_msgs/msg/bool.hpp>

// using namespace ignition;
// using namespace gazebo;
namespace gzplugin {

class LedRingPlugin : public gz::sim::System,
                     public gz::sim::ISystemConfigure, //{
                     public gz::sim::ISystemPreUpdate { 
    public:
        LedRingPlugin();
        virtual ~LedRingPlugin();

        // Implement Configure callback, provided by ISystemConfigure
        // and called once at startup.
        void Configure(const gz::sim::Entity &_entity,
                               const std::shared_ptr<const sdf::Element> &_sdf,
                               gz::sim::EntityComponentManager &_ecm,
                               gz::sim::EventManager &/*_eventMgr*/) override;
            
        // void Load(gz::sim::EntityComponentManager &_ecm, sdf::ElementPtr);

		// Update method called at every simulation step
        // void Update(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm) override;
        void PreUpdate(const gz::sim::UpdateInfo &_info,
                    gz::sim::EntityComponentManager &_ecm) override;

    private:

        void OnLedMsg(const ignition::msgs::Color &colour);
        void OnLedIndividualMsg(const ignition::msgs::Float_V &leds);

        // Node for trasnport to send to ROS2 eventually
        gz::transport::Node node;
        gz::transport::Node::Publisher led_status_pub;
        gz::transport::Node::Publisher visual_config_pub;

        // rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr dockStatusPub;
        // rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr dockControlSub;

        // rclcpp::TimerBase::SharedPtr initial_attaching_timer;
        
        gz::sim::EntityComponentManager *ecm{nullptr};
        gz::sim::EventManager *em{nullptr};
        gz::sim::Model model;
        gz::sim::World world;
        gz::sim::Link base_link;

        gz::sim::Link led_link;

        std::string publisher_name = "";
        std::string subscription_control_individual_name = "";
        std::string subscription_control_name = "";
        std::string robot_namespace = "";
        std::string visual_config_service_name = "/world/empty/visual_config";
        float ring_z_offset = -0.00; //m 
        float ring_radius = 0.1; //m
        float n_leds = 12;

        std::vector<float> led_colours;
        std::map<int, gz::sim::Entity> led_id_entity_map;

        bool _configured = false;

        bool _received = false;
    };

};

#endif  // GAZEBO_PLUGINS__GAZEBO_ROS_TEMPLATE_HPP_
