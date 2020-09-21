#include <OnnxMlirRuntime.h>
#include <stdio.h>

RtMemRefList *run_main_graph(RtMemRefList *);

int main() {
  // Construct x1 rmr filled with 1.
  float x1Data[] = {1., 1., 1., 1., 1., 1.};
  RtMemRef *x1 = rmrCreate(2);
  rmrSetData(x1, x1Data);

  // Construct x2 rmr filled with 2.
  float x2Data[] = {2., 2., 2., 2., 2., 2.};
  RtMemRef *x2 = rmrCreate(2);
  rmrSetData(x2, x2Data);

  // Construct a list of rmrs as input.
  RtMemRef *list[2] = {x1, x2};
  RtMemRefList *input = rmrListCreate(list, 2);

  // Call the compiled onnx model function.
  RtMemRefList *outputList = run_main_graph(input);

  // Get the first rmr as output.
  RtMemRef *y = rmrListGetRmrByIndex(outputList, 0);

  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++)
    printf("%f ", ((float *)rmrGetData(y))[i]);

  return 0;
}