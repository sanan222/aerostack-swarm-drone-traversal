#include "led_ring.hpp"

#include <gz/plugin/Register.hh>
// Register the plugin with Gazebo Fortress
IGNITION_ADD_PLUGIN(
    gzplugin::LedRingPlugin,
    gz::sim::System,
	gzplugin::LedRingPlugin::ISystemConfigure, //)
	gzplugin::LedRingPlugin::ISystemPreUpdate)

// Optional alias for the plugin
IGNITION_ADD_PLUGIN_ALIAS(gzplugin::LedRingPlugin, "gzplugin::LedRingPlugin")


#define PI 3.14159265358979323846

// using namespace ignition;
namespace gzplugin
{
	
// Constructor
LedRingPlugin::LedRingPlugin(){}

// Destructor
LedRingPlugin::~LedRingPlugin() {}

// void LedRingPlugin::Load(gz::sim::EntityComponentManager &_ecm, sdf::ElementPtr _sdf) {
  // Implement Configure callback, provided by ISystemConfigure
  // and called once at startup.
void LedRingPlugin::Configure(const gz::sim::Entity &_entity,
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
		ignerr << "LedRingPlugin should be attached to a model entity, failed to initialise" << std::endl;
		return;
	}

	// World is initialised in the postupdate

	// // // Get parameters specified in the sdf file.
	if (_sdf->HasElement("robotNamespace")) {
		this->robot_namespace = _sdf->Get<std::string>("robotNamespace");
	}
	if (_sdf->HasElement("ring_z_offset")) {
		this->ring_z_offset = _sdf->Get<double>("ring_z_offset");
	}
	if (_sdf->HasElement("ring_radius")) {
		this->ring_radius = _sdf->Get<double>("ring_radius");
	}
	if (_sdf->HasElement("n_leds")) {
		this->n_leds = _sdf->Get<double>("n_leds");
	}
	
	// Initialise Led Colours
	for(int i = 0; i < this->n_leds * 3; i++) {
		this->led_colours.push_back(0.0);
	}

	// Create Publisher and Subscriber for ROS2 to interact with this
	// this->publisher_name = this->robot_namespace + "/leds/status";
	// this->led_status_pub = node.Advertise<ignition::msgs::Float_V>(this->publisher_name);
	// if (!this->led_status_pub)
	// {
	// 	ignerr << "Error advertising topic [" << this->publisher_name << "]" << std::endl;
	// 	return;
	// }

	this->subscription_control_individual_name = this->robot_namespace+"/leds/control_individual";
	if(!this->node.Subscribe(this->subscription_control_individual_name, &LedRingPlugin::OnLedIndividualMsg, this))
	{
		ignerr << "Error subscribing to topic [" << this->subscription_control_individual_name << "]" << std::endl;
		return;
	}

	this->subscription_control_name = this->robot_namespace+"/leds/control";
	if(!this->node.Subscribe(this->subscription_control_name, &LedRingPlugin::OnLedMsg, this))
	{
		ignerr << "Error subscribing to topic [" << this->subscription_control_name << "]" << std::endl;
		return;
	}

	// ignerr << "Publisher initialised on " << this->subscription_name << std::endl;
	ignerr << "Subscriber initialised on " << this->subscription_control_individual_name<< std::endl;
	ignerr << "Subscriber initialised on " << this->subscription_control_name<< std::endl;
}

void LedRingPlugin::OnLedMsg(const ignition::msgs::Color &colour) {
	auto msg = ignition::msgs::Float_V();
	for(int i = 0; i < this->n_leds ; i++){
		msg.add_data(colour.r());
		msg.add_data(colour.g());
		msg.add_data(colour.b());
	}
	this->OnLedIndividualMsg(msg);
}

void LedRingPlugin::OnLedIndividualMsg(const ignition::msgs::Float_V &leds) {
	auto data = leds.data();
	ignerr << "Received an LED Control msg" << std::endl;
	if(!this->_configured) {
		ignerr << "LED Ring Not Configured Yet" << std::endl;
		return;
	}

	int data_size = data.size();
	int num_leds_in_data = (int) data_size / 3;
	if(num_leds_in_data > this->led_colours.size()) {
		ignerr << "Received more led colours than expected" << std::endl;
	}

	// for(float d: data) {
	for(int i = 0; i < num_leds_in_data; i++) {
		float r = data[i*3];
		float g = data[i*3 + 1];
		float b = data[i*3 + 2];
	
		ignerr << i << " " << std::to_string(r) + " " << std::to_string(g) + " " << std::to_string(b) << std::endl;

		// Check if different, then send request msg chaneg
		// if(this->led_colours[i*3] - r > 1e-5 ||
		// 	this->led_colours[i*3 + 1] - g > 1e-5 ||
		// 	this->led_colours[i*3 + 2] - b > 1e-5
		// ) {			
		gz::msgs::Visual msg;
		gz::sim::Entity visual_entity = this->led_id_entity_map[i];
		msg.set_id(visual_entity);

		// Modify the material color (RGBA format)
		gz::msgs::Material *mat = msg.mutable_material();
		gz::msgs::Set(mat->mutable_diffuse(), gz::math::Color(r, g, b, 1.0));  // Red color
		gz::msgs::Set(mat->mutable_emissive(), gz::math::Color(r, g, b, 1.0));  // Red color

		std::function<void(const ignition::msgs::Boolean &, const bool)> cb =
			[](const ignition::msgs::Boolean &/*_rep*/, const bool _result)
		{
			if (!_result)
			ignerr << "Error setting material color configuration"
					<< " on visual" << std::endl;
		};

		this->node.Request(this->visual_config_service_name, msg, cb);

		ignerr << "Sent Service Call Diffusive and Emmisive to " << visual_entity << std::endl;
		this->visual_config_pub.Publish(msg);
		// }

		this->led_colours[i*3] = r;
		this->led_colours[i*3 + 1] = g;
		this->led_colours[i*3 + 2] = b;
	}
	this->_received = true;
}

void LedRingPlugin::PreUpdate(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm)
{
	if (!this->_configured) {

		// Set World here as ECM is not fully initialised
		this->world = gz::sim::World(gz::sim::worldEntity(_ecm));
		this->base_link = gz::sim::Link(this->model.CanonicalLink(_ecm));

		// Create Subscriber to the visual_config with world
		std::string world_name = *this->world.Name(_ecm);
		this->visual_config_service_name = "/world/"+world_name+"/visual_config";
		ignerr << "Materials Service Call Sent To " << this->visual_config_service_name << std::endl;

		gz::sim::SdfEntityCreator entityCreator(_ecm, *(this->em));

		// Construct the LED RING
		// Create a new link
		sdf::Link sdfLink;
		sdfLink.SetName("leds_link");
		sdfLink.SetPoseRelativeTo("base_frame");
		sdfLink.SetRawPose(gz::math::Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

		// Set very small mass and inertial properties
		sdfLink.SetInertial(
			gz::math::Inertial(
				gz::math::MassMatrix3d(1e-6, gz::math::Vector3d(1e-12, 1e-12, 1e-12), 
										gz::math::Vector3d(0.0, 0.0, 0.0)),
				gz::math::Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
			)
		);

		// Create the entities using the EntityCreator
		auto linkEntity = entityCreator.CreateEntities(&sdfLink);
		if(linkEntity == gz::sim::kNullEntity) {
			ignerr << "LinkEntity failed to be created" << std::endl;
		} 
		entityCreator.SetParent(linkEntity, this->model.Entity());
		_ecm.CreateComponent(linkEntity, gz::sim::components::VisualCmd());
		this->led_link = gz::sim::Link(linkEntity);

		double angle_delta = 2*PI/this->n_leds;
		for(int i = 0; i < this->n_leds; i++) {
			// Create a visual
			sdf::Visual sdfVisual;
			sdfVisual.SetName("led" + std::to_string(i));
			sdfVisual.SetPoseRelativeTo("base_frame");

			// Calculate x, y position
			float x = this->ring_radius * std::cos(angle_delta * i);
			float y = this->ring_radius * std::sin(angle_delta * i);
			sdfVisual.SetRawPose(gz::math::Pose3d(x, y, this->ring_z_offset, 0.0, 0.0, 0.0));

			// Set cylinder geometry
			sdf::Cylinder cyl;
			cyl.SetRadius(0.015);
			cyl.SetLength(0.015);
			sdf::Geometry sdfGeom;
			sdfGeom.SetType(sdf::GeometryType::CYLINDER);
			sdfGeom.SetCylinderShape(cyl);
			sdfVisual.SetGeom(sdfGeom);

			float r = this->led_colours[i * 3 + 0];
			float b = this->led_colours[i * 3 + 1];
			float g = this->led_colours[i * 3 + 2];

			// Set material
			sdf::Material sdfMaterial;
			// Set the diffuse color to match the LED's body color
			sdfMaterial.SetDiffuse(gz::math::Color(r,g,b, 1.0));
			// Set the emissive color to simulate light emission
			sdfMaterial.SetEmissive(gz::math::Color(r,g,b, 1.0));  // Bright red
			// Set the specular color to simulate glossy reflection
			sdfMaterial.SetSpecular(gz::math::Color(0.8, 0.8, 0.8, 1.0));  // Near-white for strong highlights
			// Optionally add transparency for a translucent effect
			sdfVisual.SetTransparency(0.1);  // Slightly translucent
			sdfVisual.SetMaterial(sdfMaterial);

			// Add the visual to the link
			sdfLink.AddVisual(sdfVisual);

			// Assuming `entityCreator` is already being used to create the entity
			auto visualEntity = entityCreator.CreateEntities(&sdfVisual);
			if (visualEntity == gz::sim::kNullEntity) {
				ignerr << "Failed to create visual entity" << std::endl;
				continue;
			} 
			// Ensure it's marked as a Visual in Gazebo
			_ecm.CreateComponent(visualEntity, gz::sim::components::Visual());
			// _ecm.CreateComponent(visualEntity, gz::sim::components::VisualCmd());
			_ecm.CreateComponent(visualEntity, gz::sim::components::Material());
			this->led_id_entity_map[i] = visualEntity;
			entityCreator.SetParent(visualEntity, linkEntity);


			// // Create a light
			// sdf::Light sdfLight;
			// sdfLight.SetName("led" + std::to_string(i));
			// sdfLight.SetType(sdf::LightType::POINT);
			// sdfLight.SetDiffuse(gz::math::Color(r, g, b, 1.0));
			// sdfLight.SetSpecular(gz::math::Color(0.8, 0.8, 0.8, 1.0));  // Near-white for strong highlights
			// sdfLight.SetIntensity(1.0);
			// sdfLight.SetAttenuationRange(1.0);
			// sdfLight.SetDirection(gz::math::Vector3d(0.0, 0.0, -1.0));
			// sdfLight.SetRawPose(gz::math::Pose3d(x, y, this->ring_z_offset, 0.0, 0.0, 0.0));

			// // Add the light to the link
			// sdfLink.AddLight(sdfLight);

		}

		// OPTIONAL: Add a fixed joint to rigidly attach the link to the model
		sdf::Joint sdfJoint;
		sdfJoint.SetName("led_joint");
		sdfJoint.SetType(sdf::JointType::FIXED);
		sdfJoint.SetParentLinkName("base_link");
		sdfJoint.SetChildLinkName("leds_link");

		auto jointEntity = entityCreator.CreateEntities(&sdfJoint, true); // Needs true as it avoids dynamic resolution which will fail to find the base_link
		if(jointEntity == gz::sim::kNullEntity) {
			ignerr << "led_joint jointEntity failed to be created" << std::endl;
		}
		entityCreator.SetParent(jointEntity, this->model.Entity());

		this->_configured = true;
		ignerr << "LedRingPlugin Configured"<< std::endl;
		ignmsg << "LedRingPlugin Configured"<< std::endl;
	}
} // end of PreUpdate()

} // namespace gazebo
