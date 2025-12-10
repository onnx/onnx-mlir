#include <stdio.h>  
#include "OnnxMlirRuntime.h"  
  
// 모델 함수 선언  
extern "C" OMTensorList *run_main_graph(OMTensorList *);  
  
int main() {  
    // 입력 텐서 생성  
    int64_t shape[] = {2, 3};  
    float data[] = {1, 2, 3, 4, 5, 6};  
    OMTensor *input = omTensorCreate(data, shape, 2, ONNX_TYPE_FLOAT);  
      
    // 입력 리스트 생성  
    OMTensor *inputs[] = {input};  
    OMTensorList *inputList = omTensorListCreate(inputs, 1);  
      
    // 모델 실행  
    OMTensorList *outputList = run_main_graph(inputList);  
      
    // 결과 확인  
    OMTensor *output = omTensorListGetOmtByIndex(outputList, 0);  
    float *result = (float*)omTensorGetDataPtr(output);  
      
    printf("실행 완료\n");  
      
    // 메모리 해제  
    omTensorListDestroy(outputList);  
    omTensorListDestroy(inputList);  
      
    return 0;  
}
