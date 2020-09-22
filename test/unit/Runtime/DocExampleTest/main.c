#include <OnnxMlirRuntime.h>
#include <stdio.h>

OMTensorList *run_main_graph(OMTensorList *);

int main() {
  // Construct x1 omt filled with 1.
  float x1Data[] = {1., 1., 1., 1., 1., 1.};
  OMTensor *x1 = omTensorCreate(2);
  omTensorSetData(x1, x1Data);

  // Construct x2 omt filled with 2.
  float x2Data[] = {2., 2., 2., 2., 2., 2.};
  OMTensor *x2 = omTensorCreate(2);
  omTensorSetData(x2, x2Data);

  // Construct a list of omts as input.
  OMTensor *list[2] = {x1, x2};
  OMTensorList *input = omTensorListCreate(list, 2);

  // Call the compiled onnx model function.
  OMTensorList *outputList = run_main_graph(input);

  // Get the first omt as output.
  OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);

  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++)
    printf("%f ", ((float *)omTensorGetData(y))[i]);

  return 0;
}