#include <OnnxMlirRuntime.h>
#include <stdio.h>
#include <assert.h>

OMTensorList *run_main_graph(OMTensorList *);

OMTensorList *create_input_list() {
  // Shared shape & rank.
  int64_t shape[] = {3, 2};
  int64_t num_elements = shape[0] * shape[1];
  int64_t rank = 2;

  // Construct float arrays filled with 1s or 2s.
  float *x1Data = (float *)malloc(sizeof(float) * num_elements);
  // Check if memory is allocated for generating the data.
  if(!x1Data) return NULL;
  for (int i = 0; i < num_elements; i++)
    x1Data[i] = 1.0;
  float *x2Data = (float *)malloc(sizeof(float) * num_elements);
  // Check if memory is allocated for generating the data.
  if(!x2Data){
    free(x1Data);
    return NULL;
  }
  for (int i = 0; i < num_elements; i++)
    x2Data[i] = 2.0;

  // Use omTensorCreateWithOwnership "true" so float arrays are automatically
  // freed when the Tensors are destroyed.
  OMTensor *x1 =
      omTensorCreateWithOwnership(x1Data, shape, rank, ONNX_TYPE_FLOAT, true);
  OMTensor *x2 =
      omTensorCreateWithOwnership(x2Data, shape, rank, ONNX_TYPE_FLOAT, true);

  // Construct a TensorList using the Tensors
  OMTensor *list[2] = {x1, x2};
  return omTensorListCreate(list, 2);
}

int main() {
  // Generate input TensorList
  OMTensorList *input_list = create_input_list();
  if(!input_list){
    // Return 2 for failure to create inputs.
    return 2;
  }
  // Call the compiled onnx model function.
  OMTensorList *output_list = run_main_graph(input_list);
  if (!output_list) {
    // May inspect errno to get info about the error.
    return 1;
  }

  // Get the first tensor from output list.
  OMTensor *y = omTensorListGetOmtByIndex(output_list, 0);
  omTensorPrint("Result tensor: ", y);
  float *outputPtr = (float *)omTensorGetDataPtr(y);

  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++) {
    float f = outputPtr[i];
    if (f != 3.0) {
      fprintf(stderr, "Iteration %d: expected 3.0, got %f.\n", i, f);
      exit(100);
    }
  }
  printf("\n");

  // Destroy the list and the tensors inside of it.
  // Use omTensorListDestroyShallow if only want to destroy the list themselves.
  omTensorListDestroy(input_list);
  omTensorListDestroy(output_list);
  return 0;
}
