cmake_minimum_required(VERSION 3.8)
project(model_predictive_speed_and_steer_control_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(model_predictive_speed_and_steer_control src/model_predictive_speed_and_steer_control.cpp)
ament_target_dependencies(model_predictive_speed_and_steer_control
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  ackermann_msgs
)

target_include_directories(model_predictive_speed_and_steer_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(model_predictive_speed_and_steer_control Eigen3::Eigen)

install(TARGETS model_predictive_speed_and_steer_control
  DESTINATION lib/${PROJECT_NAME})

ament_package()
