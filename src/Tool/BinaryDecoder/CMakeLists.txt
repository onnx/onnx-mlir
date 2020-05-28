add_executable(binary-decoder BinaryDecoder.cpp)
target_link_libraries(binary-decoder
        ${LLVMSupport}
        ${LLVMDemangle}
        ${CURSES_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT})