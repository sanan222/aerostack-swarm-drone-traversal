# Templates for SDF file
SDF_TREE_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{{ model_name }}">
    <static>true</static>
    <link name="tree">
      <visual name="tree_visual">
        <geometry>
          <cylinder>
            <radius>{{ diameter / 2 }}</radius>
            <length>{{ height }}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.3 0.2 0.1 1.0</ambient>
          <diffuse>0.3 0.2 0.1 1.0</diffuse>
        </material>
      </visual>
      <collision name="tree_collision">
        <geometry>
          <cylinder>
            <radius>{{ diameter / 2 }}</radius>
            <length>{{ height }}</length>
          </cylinder>
        </geometry>
      </collision>
      <pose>0 0 {{ height / 2 }} 0 0 0</pose>
    </link>
  </model>
</sdf>
"""

SDF_FLOOR_TILE_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{{ model_name }}">
    <static>true</static>
    <link name="floor_tile">
      <visual name="tile_visual">
        <geometry>
          <box>
            <size>{{ width }} {{ length }} 0.01</size>
          </box>
        </geometry>
        <material>
          <ambient>{{ colour }}</ambient>
          <diffuse>{{ colour }}</diffuse>
        </material>
      </visual>
      <pose>0 0 0.005 0 0 0</pose>
    </link>
  </model>
</sdf>
"""

SDF_DYNAMIC_OBJECT_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{{ model_name }}">
    <static>false</static>
    <link name="dynamic_object">
      <inertial>
        <mass>100</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>{{ width }} {{ length }} {{ height }}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{{ width }} {{ length }} {{ height }}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.3 0.5 1.0</ambient>
          <diffuse>0.7 0.3 0.5 1.0</diffuse>
        </material>
      </visual>
    </link>
    <plugin filename="libDynamicMovingObjects" name="gzplugin::DynamicMovingObjects">
      <boundary_min_x>{{ boundary_min_x }}</boundary_min_x>
      <boundary_max_x>{{ boundary_max_x }}</boundary_max_x>
      <boundary_min_y>{{ boundary_min_y }}</boundary_min_y>
      <boundary_max_y>{{ boundary_max_y }}</boundary_max_y>
      <movement_velocity>{{ movement_velocity }}</movement_velocity>
      <initial_angle>{{ initial_angle }}</initial_angle>
    </plugin>
  </model>
</sdf>
"""

# Template for SDF Window file
SDF_WINDOW_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{{ model_name }}">
    <static>true</static>

    <!-- Bottom Wall (below window) -->
    <link name="bottom_wall">
      <visual name="bottom_visual">
        <geometry>
          <box>
            <size>{{ room_width }} {{ wall_depth }} {{ window_bottom }}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1.0</ambient>
          <diffuse>0.5 0.5 0.5 1.0</diffuse>
        </material>
      </visual>
      <collision name="bottom_collision">
        <geometry>
          <box>
            <size>{{ room_width }} {{ wall_depth }} {{ window_bottom }}</size>
          </box>
        </geometry>
      </collision>
      <pose>0 0 {{ window_bottom / 2 }} 0 0 0</pose>
    </link>

    <!-- Top Wall (above window) -->
    <link name="top_wall">
      <visual name="top_visual">
        <geometry>
          <box>
            <size>{{ room_width }} {{ wall_depth }} {{ wall_top }}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1.0</ambient>
          <diffuse>0.5 0.5 0.5 1.0</diffuse>
        </material>
      </visual>
      <collision name="top_collision">
        <geometry>
          <box>
            <size>{{ room_width }} {{ wall_depth }} {{ wall_top }}</size>
          </box>
        </geometry>
      </collision>
      <pose>0 0 {{ window_bottom + window_height + (wall_top / 2) }} 0 0 0</pose>
    </link>

    <!-- Left Wall (left of window) -->
    <link name="left_wall">
      <visual name="left_visual">
        <geometry>
          <box>
            <size>{{ left_width }} {{ wall_depth }} {{ room_height }}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1.0</ambient>
          <diffuse>0.5 0.5 0.5 1.0</diffuse>
        </material>
      </visual>
      <collision name="left_collision">
        <geometry>
          <box>
            <size>{{ left_width }} {{ wall_depth }} {{ room_height }}</size>
          </box>
        </geometry>
      </collision>
      <pose>{{ left_x }} 0 {{ room_height / 2 }} 0 0 0</pose>
    </link>

    <!-- Right Wall (right of window) -->
    <link name="right_wall">
      <visual name="right_visual">
        <geometry>
          <box>
            <size>{{ right_width }} {{ wall_depth }} {{ room_height }}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1.0</ambient>
          <diffuse>0.5 0.5 0.5 1.0</diffuse>
        </material>
      </visual>
      <collision name="right_collision">
        <geometry>
          <box>
            <size>{{ right_width }} {{ wall_depth }} {{ room_height }}</size>
          </box>
        </geometry>
      </collision>
      <pose>{{ right_x }} 0 {{ room_height / 2 }} 0 0 0</pose>
    </link>

  </model>
</sdf>
"""


# Template for model.config
MODEL_CONFIG_TEMPLATE = """<?xml version="1.0" ?>
<model>
  <name>{{ model_name }}</name>
  <version>1.0</version>
  <sdf version="1.6">{{ model_name }}.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>
    {{ model_name }}
  </description>
</model>
"""