#include <OnnxMlirRuntime.h>
#include <iostream>
#include <dlfcn.h>

int main() {
    // 동적 라이브러리 로드
    void* handle = dlopen("./test_linalg.onnx.so", RTLD_LAZY);
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

    // 올바른 입력 텐서 생성: MatMul(2x3) * (3x4) = (2x4)
    // 입력 A: 2x3 = 6개 요소
    int64_t shapeA[] = {2, 3};
    float x1Data[] = {1.0, 2.0, 3.0,  // 첫 번째 행
                      4.0, 5.0, 6.0}; // 두 번째 행
    
    // 입력 B: 3x4 = 12개 요소
    int64_t shapeB[] = {3, 4};
    float x2Data[] = {1.0, 2.0, 3.0, 4.0,   // 첫 번째 행
                      5.0, 6.0, 7.0, 8.0,   // 두 번째 행
                      9.0, 10.0, 11.0, 12.0}; // 세 번째 행
    
    OMTensor *x1 = omTensorCreate(x1Data, shapeA, 2, ONNX_TYPE_FLOAT);
    OMTensor *x2 = omTensorCreate(x2Data, shapeB, 2, ONNX_TYPE_FLOAT);
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

    // 결과 확인: 출력은 2x4 = 8개 요소
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

    // 정리
    omTensorListDestroy(output);
    omTensorListDestroy(input);
    dlclose(handle);

    return 0;
}

