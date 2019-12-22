add_library(cruntime
        dyn_memref.cpp
        dyn_memref.h
        data_type.h)
target_include_directories(cruntime
        PRIVATE ${DLC_SRC_ROOT} ${DLC_BIN_ROOT}
        ${DLC_SRC_ROOT})

pybind11_add_module(pyruntime
        dyn_memref.cpp
        dyn_memref.h
        runtime.cpp
        runtime.hpp)
target_link_libraries(pyruntime PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(pyruntime
        PRIVATE ${ONNF_SRC_ROOT} ${ONNF_BIN_ROOT}
        ${ONNF_SRC_ROOT})
