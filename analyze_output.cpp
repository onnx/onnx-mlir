#include <OnnxMlirRuntime.h>
#include <iostream>
#include <dlfcn.h>

int main() {
    void* handle = dlopen("./test_linalg3.onnx.so", RTLD_LAZY);
    if (!handle) return 1;

    typedef OMTensorList* (*RunFunc)(OMTensorList*);
    RunFunc runFunc = (RunFunc)dlsym(handle, "run_main_graph");
    if (!runFunc) return 1;

    // 입력 텐서 생성
    int64_t shape[] = {2, 4};
    float x1Data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float x2Data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    
    OMTensor *x1 = omTensorCreate(x1Data, shape, 2, ONNX_TYPE_FLOAT);
    OMTensor *x2 = omTensorCreate(x2Data, shape, 2, ONNX_TYPE_FLOAT);
    OMTensor *list[] = {x1, x2};
    OMTensorList *input = omTensorListCreate(list, 2);

    // 입력 확인
    std::cout << "Input 1 shape: ";
    const int64_t* s1 = omTensorGetShape(x1);
    for (int i = 0; i < omTensorGetRank(x1); i++) {
        std::cout << s1[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Input 1 data: ";
    float* d1 = (float*)omTensorGetDataPtr(x1);
    for (int i = 0; i < 8; i++) {
        std::cout << d1[i] << " ";
    }
    std::cout << std::endl;

    // 모델 실행
    OMTensorList *output = runFunc(input);

    // 출력 확인
    OMTensor *y = omTensorListGetOmtByIndex(output, 0);
    std::cout << "Output shape: ";
    const int64_t* sy = omTensorGetShape(y);
    for (int i = 0; i < omTensorGetRank(y); i++) {
        std::cout << sy[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output data: ";
    float *outputData = (float*)omTensorGetDataPtr(y);
    int64_t numElements = 1;
    for (int i = 0; i < omTensorGetRank(y); i++) {
        numElements *= sy[i];
    }
    for (int i = 0; i < numElements; i++) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    omTensorListDestroy(output);
    omTensorListDestroy(input);
    dlclose(handle);
    return 0;
}
