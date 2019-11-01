
add_executable(onnf main.cpp)
target_include_directories(onnf PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnf PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(onnf builder compiler ${Boost_LIBRARIES})
