It is often the case where we want to know two dynamic dimensions are equal or not at compile time. This helps with decision on how to lowering an ONNX operator. For example, given an ONNXAddOp as follows:

```mlir
%0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3x5xf32>, tensor<?x3x5xf32>) -> tensor<?x3x5xf32>
```
If we know at compile time that the first dimensions of `%arg0` and `%arg1` are the same (e.g., coming from the same tensor), there is no need to generate runtime code to handle broadcasting rules.

This also helps generate code for accelerators. If an accelerator does not support broadcasting, we can check at compile to decide whether the ONNXAddOp will be offloaded to the accelerator or not.

We provide a helper class [DimAnalysis](../src/Transform/ONNX/ONNXDimAnalysis.hpp) to analyze dynamic dimensions and to check whether two dynamic dimensions are the same or not. Below is an example of using DimAnalysis:

```C
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

// Run the dynamic dimension analysis to help check equality of dynamic
// dimensions at compile time.
ModuleOp moduleOp = getOperation();
onnx_mlir::DimAnalysis dimAnalysis(moduleOp);
dimAnalysis.analyze();
```

DimAnalysis is constructed for a ModuleOp so that all operations in the ModuleOp will be analyzed.
Then, actual analysis is done via calling `analyze()` function.
After that, we can query if two dynamic dimensions are the same or not via calling
```C
bool sameDim = dimAnalysis.sameDynDim(tensor1, dimAxis1, tensor2, dimAxis2);
```
where the first dynamic dimension is identified by its tensor `tensor1` and its axis `dimAxis1`, and the second dynamic dimension by `tensor2` and `dimAxis2`.

DimAnalysis has been using for NNPA, please see [ONNXToZHigh](../src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.cpp) for more information.
