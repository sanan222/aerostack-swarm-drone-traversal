#pragma once
#ifndef GAZEBO_PLUGINS__DYNAMIC_MOVING_OBJECT_HPP_
#define GAZEBO_PLUGINS__DYNAMIC_MOVING_OBJECT_HPP_

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


namespace gzplugin {

class DynamicMovingObjects : public gz::sim::System,
                            public gz::sim::ISystemConfigure, //{
                            public gz::sim::ISystemPreUpdate { 
    public:
        DynamicMovingObjects();
        virtual ~DynamicMovingObjects();

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

        // Node for trasnport to send to ROS2 eventually
        gz::transport::Node node;
        gz::transport::Node::Publisher location_pub;

        gz::sim::EntityComponentManager *ecm{nullptr};
        gz::sim::EventManager *em{nullptr};
        gz::sim::Model model;
        gz::sim::World world;
        gz::sim::Link base_link;

        std::string topic = "/dynamic_obstacles/locations";
        std::string pose_service_name = "";

        double boundary_min_x = -2.0;
        double boundary_max_x = -2.0;
        double boundary_min_y = 2.0;
        double boundary_max_y = 2.0;

        float obj_size_x = 0.5;
        float obj_size_y = 0.5;
        float obj_size_z = 3.0;
        
        double movement_velocity = 0.5;
        double initial_angle = 0.0; // radians
        gz::math::Vector3d velocity_direction; 

        double topic_hz = 3.0;
        double current_time = 0.0;

        bool _configured = false;
    };

};

#endif  // GAZEBO_PLUGINS__DYNAMIC_MOVING_OBJECT_HPP_
