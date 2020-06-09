add_executable(binary-decoder BinaryDecoder.cpp)
target_link_libraries(binary-decoder
        ${LLVMSupport}
        ${LLVMDemangle}
        ${CURSES_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT})

message(STATUS "incl dir" ${ONNX_INCLUDE_DIRS})
target_link_libraries(binary-decoder onnx)