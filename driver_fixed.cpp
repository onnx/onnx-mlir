#include <OnnxMlirRuntime.h>
#include <iostream>
#include <dlfcn.h>

int main() {
    // 동적 라이브러리 로드
    void* handle = dlopen("./test_krnl.onnx.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load library: " << dlerror() << std::endl;
        return 1;
    }

    // omQueryEntryPoints 함수 찾기
    typedef const char** (*QueryEntryPointsFunc)(int64_t*);
    QueryEntryPointsFunc queryEntryPoints = (QueryEntryPointsFunc)dlsym(handle, "omQueryEntryPoints");
    if (!queryEntryPoints) {
        std::cerr << "Cannot load symbol omQueryEntryPoints: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    // Entry point 확인
    int64_t numEntryPoints = 0;
    const char** entryPoints = queryEntryPoints(&numEntryPoints);
    std::cout << "Found " << numEntryPoints << " entry point(s)" << std::endl;

    // 입력 텐서 생성
    int64_t shape[] = {2, 4};
    float x1Data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float x2Data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    
    OMTensor *x1 = omTensorCreate(x1Data, shape, 2, ONNX_TYPE_FLOAT);
    OMTensor *x2 = omTensorCreate(x2Data, shape, 2, ONNX_TYPE_FLOAT);
    OMTensor *list[] = {x1, x2};
    OMTensorList *input = omTensorListCreate(list, 2);

    // run_main_graph 함수 찾기
    typedef OMTensorList* (*RunFunc)(OMTensorList*);
    RunFunc runFunc = (RunFunc)dlsym(handle, "run_main_graph");
    if (!runFunc) {
        std::cerr << "Cannot load symbol run_main_graph: " << dlerror() << std::endl;
        omTensorListDestroy(input);
        dlclose(handle);
        return 1;
    }

    // 모델 실행
    OMTensorList *output = runFunc(input);

    // 결과 확인
    OMTensor *y = omTensorListGetOmtByIndex(output, 0);
    float *outputData = (float*)omTensorGetDataPtr(y);
    
    std::cout << "Output: ";
    for (int i = 0; i < 8; i++) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // 정리
    omTensorListDestroy(output);
    omTensorListDestroy(input);
    dlclose(handle);

    return 0;
}
