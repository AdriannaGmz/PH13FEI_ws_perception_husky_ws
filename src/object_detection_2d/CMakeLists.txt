cmake_minimum_required(VERSION 2.8.3)
project(object_detection_2d)

add_compile_options(-std=c++11)

set(PROJECT_DEPS
        roscpp
        rospy
        std_msgs
        cv_bridge
        message_generation
        )

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        cv_bridge
        message_generation
        )

find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")

add_message_files(DIRECTORY msg
        FILES
        BoundingBox2D.msg
        BoundingBoxes.msg
)

catkin_python_setup()

generate_messages(
       DEPENDENCIES
       std_msgs
       object_detection_2d
)

catkin_package(CATKIN_DEPENDS roscpp rospy message_runtime std_msgs)


catkin_install_python(PROGRAMS src/main.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${roscpp_INCLUDE_DIRS}
        ${rospy_INCLUDE_DIRS}
)

#add_executable(object_detection_2d)
#target_link_libraries(object_detection_2d ${catkin_LIBRARIES} ${OpenCV_LIBS})