#include <OnnxMlirRuntime.h>
#include <stdio.h>

OMTensorList *run_main_graph(OMTensorList *);

int main() {
  // Shared shape & rank.
  int64_t shape[] = {2, 2};
  int64_t rank = 2;
  // Construct x1 omt filled with 1.
  float x1Data[] = {1., 1., 1., 1., 1., 1.};
  int64_t *x1Shape = {2, 2};
  OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);
  // Construct x2 omt filled with 2.
  float x2Data[] = {2., 2., 2., 2., 2., 2.};
  int64_t *x2Shape = {2, 2};
  OMTensor *x2 = omTensorCreate(x2Data, shape, rank, ONNX_TYPE_FLOAT);
  // Construct a list of omts as input.
  OMTensor *list[2] = {x1, x2};
  OMTensorList *input = omTensorListCreate(list, 2);
  // Call the compiled onnx model function.
  OMTensorList *outputList = run_main_graph(input);
  // Get the first omt as output.
  OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);
  float *outputPtr = (float *)omTensorGetAlignedPtr(y);
  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++)
    printf("%f ", outputPtr[i]);
  return 0;
}