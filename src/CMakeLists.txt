
add_executable(onnf main.cpp)

target_include_directories(onnf PRIVATE ..)

target_link_libraries(onnf
                      builder
                      ${Boost_LIBRARIES}
                      )

