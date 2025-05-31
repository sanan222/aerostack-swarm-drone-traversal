#include "dynamic_object.hpp"

#include <gz/plugin/Register.hh>
// Register the plugin with Gazebo Fortress
IGNITION_ADD_PLUGIN(
    gzplugin::DynamicMovingObjects,
    gz::sim::System,
	gzplugin::DynamicMovingObjects::ISystemConfigure, //)
	gzplugin::DynamicMovingObjects::ISystemPreUpdate)

// Optional alias for the plugin
IGNITION_ADD_PLUGIN_ALIAS(gzplugin::DynamicMovingObjects, "gzplugin::DynamicMovingObjects")


#define PI 3.14159265358979323846

// using namespace ignition;
namespace gzplugin
{
	
// Constructor
DynamicMovingObjects::DynamicMovingObjects(){}

// Destructor
DynamicMovingObjects::~DynamicMovingObjects() {}

// void DynamicMovingObjects::Load(gz::sim::EntityComponentManager &_ecm, sdf::ElementPtr _sdf) {
  // Implement Configure callback, provided by ISystemConfigure
  // and called once at startup.
void DynamicMovingObjects::Configure(const gz::sim::Entity &_entity,
				const std::shared_ptr<const sdf::Element> &_sdf,
				gz::sim::EntityComponentManager &_ecm,
				gz::sim::EventManager &_eventMgr)
{

	// // Create a GazeboRos node instead of a common ROS node.
	// // Pass it SDF parameters so common options like namespace and remapping
	// // can be handled. 
	// Not needed anymore as ROS not integrated as closely
	// ros_node_ = gazebo_ros::Node::Get(_sdf);

	this->ecm = &_ecm;
	this->em = &_eventMgr;

	// // Obtain the model entity (model pointer is no longer passed directly)
    // gz::sim::Entity modelEntity = gz::sim::Model(_sdf->GetParent());
    this->model = gz::sim::Model(_entity); //std::make_shared<gz::physics::Model>(modelEntity);
	if(!this->model.Valid(_ecm))
	{
		ignerr << "Dynamic Object should be attached to a model entity, failed to initialise" << std::endl;
		return;
	}

	// World is initialised in the postupdate

	// // // Get parameters specified in the sdf file.
	if (_sdf->HasElement("topic")) {
		this->topic = _sdf->Get<std::string>("topic");
	}
	if (_sdf->HasElement("topic_hz")) {
		this->topic_hz = _sdf->Get<double>("topic_hz");
	}

	if (_sdf->HasElement("boundary_min_x")) {
		this->boundary_min_x = _sdf->Get<double>("boundary_min_x");
	}
	if (_sdf->HasElement("boundary_min_y")) {
		this->boundary_min_y = _sdf->Get<double>("boundary_min_y");
	}
	if (_sdf->HasElement("boundary_max_x")) {
		this->boundary_max_x = _sdf->Get<double>("boundary_max_x");
	}
	if (_sdf->HasElement("boundary_max_y")) {
		this->boundary_max_y = _sdf->Get<double>("boundary_max_y");
	}

	if (_sdf->HasElement("obj_size_x")) {
		this->obj_size_x = _sdf->Get<float>("obj_size_x");
	}
	if (_sdf->HasElement("obj_size_y")) {
		this->obj_size_y = _sdf->Get<float>("obj_size_y");
	}
	if (_sdf->HasElement("obj_size_z")) {
		this->obj_size_z = _sdf->Get<float>("obj_size_z");
	}
	
	if (_sdf->HasElement("movement_velocity")) {
		this->movement_velocity = _sdf->Get<double>("movement_velocity");
	}

	if (_sdf->HasElement("initial_angle")) {
		this->initial_angle = _sdf->Get<double>("initial_angle");
	}

	this->velocity_direction = this->movement_velocity * gz::math::Vector3d(cos(this->initial_angle), sin(this->initial_angle), 0);  // Initial movement along X-axis
	
	// Create Publisher and Subscriber for ROS2 to interact with this
	this->location_pub = node.Advertise<ignition::msgs::Pose>(this->topic);
	if (!this->location_pub)
	{
		ignerr << "Error advertising topic [" << this->topic << "]" << std::endl;
		return;
	} 

	ignerr << "Publisher initialised on " << this->topic<< std::endl;
}

void DynamicMovingObjects::PreUpdate(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm)
{
	if (!this->_configured) {
		this->world = gz::sim::World(gz::sim::worldEntity(_ecm));
		// Create Subscriber to the visual_config with world
		std::string world_name = *this->world.Name(_ecm);
		this->pose_service_name = "/world/"+world_name+"/set_pose";
		
		this->_configured = true;
	}

	if (_info.paused)
        return;

	auto poseComp = _ecm.Component<gz::sim::components::Pose>(this->model.Entity());
	if (!poseComp)
	{
		ignerr << "Pose component missing from model." << std::endl;
		return;
	}

	gz::math::Pose3d pose = poseComp->Data();

	// Compute time step (convert nanoseconds to seconds)
	double dt = std::chrono::duration<double>(_info.dt).count();

	// Compute new position
	gz::math::Vector3d new_position = pose.Pos() + this->velocity_direction * dt;

	// Check boundary conditions and reverse direction if needed
	if (new_position.X() < boundary_min_x || new_position.X() > boundary_max_x)
	{
	this->velocity_direction.X(-this->velocity_direction.X());
	new_position.X(std::clamp(new_position.X(), boundary_min_x, boundary_max_x));
	}
	if (new_position.Y() < boundary_min_y || new_position.Y() > boundary_max_y)
	{
	this->velocity_direction.Y(-this->velocity_direction.Y());
	new_position.Y(std::clamp(new_position.Y(), boundary_min_y, boundary_max_y));
	}

	// Apply new position
	// pose.SetX(new_position.X());
	// pose.SetY(new_position.Y());

	// Set updated pose component
	// _ecm.SetComponentData<gz::sim::components::Pose>(this->model.Entity(), pose);

	// Call service
	gz::msgs::Pose msg;
	msg.set_id(this->model.Entity());
	gz::msgs::Header *header = msg.mutable_header();
	header->mutable_stamp()->set_sec(_info.simTime.count() / 1000000000);  // Convert nanoseconds to seconds
	header->mutable_stamp()->set_nsec(_info.simTime.count() % 1000000000);
	auto frame = header->add_data();
	frame->set_key("frame_id");
	frame->add_value("object_" + std::to_string(this->model.Entity()));
	msg.set_name("object_" + std::to_string(this->model.Entity()));
	gz::msgs::Vector3d *vec = msg.mutable_position();
	vec->set_x(new_position.X());
	vec->set_y(new_position.Y());
	
	std::function<void(const ignition::msgs::Boolean &, const bool)> cb =
	[](const ignition::msgs::Boolean &/*_rep*/, const bool _result)
	{
		if (!_result)
		ignerr << "Error setting pose configuration" << std::endl;
	};

	this->node.Request(this->pose_service_name, msg, cb);

	// Location Pub limited to topic_hz
	this->current_time = this->current_time + dt;
	// Check if enough time has passed since last publish (0.1s for 10Hz)
	if (current_time  >= (1.0 / this->topic_hz))
	{
		this->location_pub.Publish(msg);
		this->current_time = 0.0;
	}

	
} // end of PreUpdate()

} // namespace gazebo
