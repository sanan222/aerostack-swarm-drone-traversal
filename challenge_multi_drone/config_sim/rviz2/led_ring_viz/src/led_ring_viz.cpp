#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int16_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>
#include <string>
#include <vector>

class LEDRingVisualizer : public rclcpp::Node {
public:
    LEDRingVisualizer() : Node("led_ring_visualizer"),  tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)  {
        // Declare and get parameters
        this->declare_parameter<std::vector<std::string>>("namespace_list", {});
        robot_namespaces_ = this->get_parameter("namespace_list").as_string_array();

        this->declare_parameter<double>("z_offset", 0.00);
        z_offset = this->get_parameter("z_offset").as_double();

        this->declare_parameter<double>("ring_radius", 0.1);
        ring_radius = this->get_parameter("ring_radius").as_double();

        this->declare_parameter<int>("num_leds", 12);
        num_leds = this->get_parameter("num_leds").as_int();

        this->declare_parameter<double>("publish_frequency", 20.0);
        publish_frequency_ = this->get_parameter("publish_frequency").as_double();

        this->declare_parameter<std::string>("frame_id", "earth");
        frame_id = this->get_parameter("frame_id").as_string();


        // Initialize subscribers and publisher
        for (const auto &namespace_ : robot_namespaces_) {
            std::string topic = "/" + namespace_ + "/leds";

            std_msgs::msg::UInt16MultiArray::SharedPtr msg = std::make_shared<std_msgs::msg::UInt16MultiArray>();
            this->led_data[namespace_] = msg;

            auto callback = [this, namespace_](std_msgs::msg::UInt16MultiArray::SharedPtr msg) {
                this->led_data[namespace_] = msg;
            };
            subscribers_.push_back(this->create_subscription<std_msgs::msg::UInt16MultiArray>(topic, 10, callback));
        
            marker_publishers_[namespace_] = this->create_publisher<visualization_msgs::msg::Marker>("viz/" + namespace_ + "/led", 10);
        }

        // Initialize timer
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_frequency_),
            std::bind(&LEDRingVisualizer::publish_visualizations, this));

        RCLCPP_INFO(this->get_logger(), "LED Ring Visualiser Initialised");
    }

private:
    void publish_visualizations() {

        for (const auto &namespace_msg_pair : led_data) {
            const std::string &namespace_ = namespace_msg_pair.first;
            const auto &msg = namespace_msg_pair.second;

            const auto &data = msg->data;
            size_t num_leds_packet = data.size() / 3;
            if (data.size() % 3 != 0) {
                RCLCPP_WARN(this->get_logger(), "Malformed LED data for namespace %s", namespace_.c_str());
                return;
            }
            if(num_leds_packet > this->num_leds) {
                RCLCPP_WARN(this->get_logger(), "More LED values (%ld) than expected (%d)", num_leds_packet, this->num_leds);
            }

            geometry_msgs::msg::TransformStamped transform;
            try {
                transform = tf_buffer_.lookupTransform("earth", namespace_, tf2::TimePointZero);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Could not transform from %s to earth: %s", namespace_.c_str(), ex.what());
                continue;
            }

            // Generate a single "POINTS" marker for LED ring
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = this->frame_id;
            marker.header.stamp = this->get_clock()->now();
            marker.ns = namespace_;
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.02; // Point size in X direction
            marker.scale.y = 0.02; // Point size in Y direction
            marker.color.a = 1.0; // Ensure points are visible
            // marker.frame_locked=true;

            for (size_t i = 0; i < this->num_leds; ++i) {
                
                float r = 0.0, g = 0.0, b = 0.0;
                if(i < num_leds_packet) {
                    r = data[i * 3] / 255.0f;
                    g = data[i * 3 + 1] / 255.0f;
                    b = data[i * 3 + 2] / 255.0f;
                } 
                

                double angle = 2.0 * M_PI * i / this->num_leds;
                double x = this->ring_radius * cos(angle);
                double y = this->ring_radius * sin(angle);
                double z = this->z_offset;


                geometry_msgs::msg::Point local_point;
                local_point.x = x;
                local_point.y = y;
                local_point.z = z;

                geometry_msgs::msg::Point transformed_point;
                tf2::doTransform(local_point, transformed_point, transform);
                marker.points.push_back(transformed_point);

                // geometry_msgs::msg::Point point;
                // point.x = x;
                // point.y = y;
                // point.z = z;
                // marker.points.push_back(point);

                std_msgs::msg::ColorRGBA color;
                color.r = r;
                color.g = g;
                color.b = b;
                color.a = 1.0;
                marker.colors.push_back(color);
            }

            // Publish the single marker
            marker_publishers_[namespace_]->publish(marker);
            // Publish markers
            // marker_publishers_[namespace_]->publish(marker_array);
        }
    }

    float z_offset = -0.1;
    float ring_radius = 0.5;
    int num_leds = 12;
    double publish_frequency_;
    std::string frame_id;

    std::vector<std::string> robot_namespaces_;

    std::vector<rclcpp::Subscription<std_msgs::msg::UInt16MultiArray>::SharedPtr> subscribers_;
    std::map<std::string, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr> marker_publishers_;

    std::map<std::string, std_msgs::msg::UInt16MultiArray::SharedPtr> led_data;

    rclcpp::TimerBase::SharedPtr timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;


};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LEDRingVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
