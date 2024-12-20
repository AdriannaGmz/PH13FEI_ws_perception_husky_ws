cmake_minimum_required(VERSION 2.8.3)
project(object_detection_depth)

add_compile_options(-std=c++11)

set(PROJECT_DEPS 
roscpp
rospy
std_msgs
cv_bridge
sensor_msgs
geometry_msgs
message_generation
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  geometry_msgs
  object_detection_2d
  message_generation
)
find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")


add_message_files(DIRECTORY msg
  FILES
  CustomPosewithCovarianceStamped.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
  object_detection_2d
  object_detection_depth
)

catkin_package(CATKIN_DEPENDS 
roscpp rospy message_runtime sensor_msgs geometry_msgs std_msgs
object_detection_2d)

catkin_install_python(PROGRAMS src/main.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${roscpp_INCLUDE_DIRS}
        ${rospy_INCLUDE_DIRS}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_object_detection_depth.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
