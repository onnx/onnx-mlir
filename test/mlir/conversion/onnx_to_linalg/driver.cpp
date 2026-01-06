#include <OnnxMlirRuntime.h>
#include <iostream>
#include <dlfcn.h>

int main() {
    // Load dynamic library
    void* handle = dlopen("./test_linalg.onnx.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load library: " << dlerror() << std::endl;
        return 1;
    }

    // Find omQueryEntryPoints function
    typedef const char** (*QueryEntryPointsFunc)(int64_t*);
    QueryEntryPointsFunc queryEntryPoints = (QueryEntryPointsFunc)dlsym(handle, "omQueryEntryPoints");
    if (!queryEntryPoints) {
        std::cerr << "Cannot load symbol omQueryEntryPoints: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Check entry points
    int64_t numEntryPoints = 0;
    const char** entryPoints = queryEntryPoints(&numEntryPoints);
    std::cout << "Found " << numEntryPoints << " entry point(s)" << std::endl;

    // Create correct input tensors: MatMul(2x3) * (3x4) = (2x4)
    // Input A: 2x3 = 6 elements
    int64_t shapeA[] = {2, 3};
    float x1Data[] = {1.0, 2.0, 3.0,  // First row
                      4.0, 5.0, 6.0}; // Second row
    
    // Input B: 3x4 = 12 elements
    int64_t shapeB[] = {3, 4};
    float x2Data[] = {1.0, 2.0, 3.0, 4.0,   // First row
                      5.0, 6.0, 7.0, 8.0,   // Second row
                      9.0, 10.0, 11.0, 12.0}; // Third row
    
    OMTensor *x1 = omTensorCreate(x1Data, shapeA, 2, ONNX_TYPE_FLOAT);
    OMTensor *x2 = omTensorCreate(x2Data, shapeB, 2, ONNX_TYPE_FLOAT);
    OMTensor *list[] = {x1, x2};
    OMTensorList *input = omTensorListCreate(list, 2);

    // Find run_main_graph function
    typedef OMTensorList* (*RunFunc)(OMTensorList*);
    RunFunc runFunc = (RunFunc)dlsym(handle, "run_main_graph");
    if (!runFunc) {
        std::cerr << "Cannot load symbol run_main_graph: " << dlerror() << std::endl;
        omTensorListDestroy(input);
        dlclose(handle);
        return 1;
    }

    // Execute model
    OMTensorList *output = runFunc(input);

    // Check results: output is 2x4 = 8 elements
    OMTensor *y = omTensorListGetOmtByIndex(output, 0);
    float *outputData = (float*)omTensorGetDataPtr(y);
    
    std::cout << "Expected result (2x4):" << std::endl;
    std::cout << "  Row 0: 38, 44, 50, 56" << std::endl;
    std::cout << "  Row 1: 83, 98, 113, 128" << std::endl;
    std::cout << std::endl;
    std::cout << "Actual output:" << std::endl;
    for (int i = 0; i < 2; i++) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < 4; j++) {
            std::cout << outputData[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    omTensorListDestroy(output);
    omTensorListDestroy(input);
    dlclose(handle);

    return 0;
}

