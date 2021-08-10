# example 01: contact
add_executable(contact ${CMAKE_CURRENT_SOURCE_DIR}/01_contact.cpp)
target_link_libraries(contact PUBLIC PFNN)

# # example 02: posedata
# add_executable(posedata ${CMAKE_CURRENT_SOURCE_DIR}/02_posedata.cpp)
# target_link_libraries(posedata PUBLIC PFNN)

# # example 03: visualization_ex
# add_executable(visualization ${CMAKE_CURRENT_SOURCE_DIR}/03_visualization_ex.cpp)
# target_link_libraries(visualization PUBLIC PFNN)

# # example 04: FCnet_train
# add_executable(fctrain ${CMAKE_CURRENT_SOURCE_DIR}/04_FCnet_train.cpp)
# target_link_libraries(fctrain PUBLIC PFNN)

# # example 05: FCnet_test
# add_executable(fctest ${CMAKE_CURRENT_SOURCE_DIR}/05_FCnet_test.cpp)
# target_link_libraries(fctest PUBLIC PFNN)

# # example 06: reconstruct_pose
# add_executable(reconstruct ${CMAKE_CURRENT_SOURCE_DIR}/06_reconstruct_pose.cpp)
# target_link_libraries(reconstruct PUBLIC PFNN)

# # example 07: pfnn_train
# add_executable(pfnntrain ${CMAKE_CURRENT_SOURCE_DIR}/07_pfnn_train.cpp)
# target_link_libraries(pfnntrain PUBLIC PFNN)

# # example 08: pfnn_test
# add_executable(pfnntest ${CMAKE_CURRENT_SOURCE_DIR}/08_pfnn_test.cpp)
# target_link_libraries(pfnntest PUBLIC PFNN)
