add_library(PFNN SHARED 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/contactlabel.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/phaselabel.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/posedata.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/roottrajectory.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/tensordata.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp

)

# headers
target_include_directories(PFNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# link aOpenGL, aLibTorch
target_link_libraries(PFNN PUBLIC aLibTorch aOpenGL)



# # PFNN library 만들기
# add_library(PFNN SHARED 
#     ${CMAKE_CURRENT_SOURCE_DIR}/code/src/pfnn.cpp)

# target_include_directories(PFNN PUBLIC 
#     ${CMAKE_CURRENT_SOURCE_DIR}/code/include)

# target_link_libraries(PFNN PUBLIC aLibTorch aOpenGL)

# # PFNN library를 사용하는 test code
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)