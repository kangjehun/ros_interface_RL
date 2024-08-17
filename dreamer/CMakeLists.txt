cmake_minimum_required(VERSION 3.0.2)
project(dreamer)

find_package(catkin REQUIRED COMPONENTS
  # ackermann_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES dreamer
#  CATKIN_DEPENDS ackermann_msgs rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)