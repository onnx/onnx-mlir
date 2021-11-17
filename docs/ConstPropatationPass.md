# Constant Propagation for ONNX operations

This document describes the `--constprop-onnx` pass which is used to do
constant propagation for operations in the ONNX dialect.

## Example
Given the following code:
```mlir
func @foo() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "std.return"(%2) : (tensor<1xf32>) -> ()
}
```

If we call `onnx-mlir-op --constprop-onnx`, we get:
```mlir
func @foo() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[3.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  "std.return"(%0) : (tensor<1xf32>) -> ()
}
```

## Remark 

ONNXConstantOp uses DenseElementsAttr to store constant values. It is
important to note that, once a DenseElementsAttr is created, it is alive and 
consumes memory until the end of compilation. In [Example](#example), all the
three DenseElementsAttrs in the three ONNXConstantOps (two in the
original code and one in the result code) exist until the end of compilation. 

To avoid creating too many DenseElementsAttrs for intermediate ONNXConstant ops
during `--constprop-onnx`, we design a mechanism that dynamically allocates and
deallocates buffers for intermediate ONNXConstantOps and only creates
DenseElementsAttr for the final result of constant propagation.

In particular, we maintain a buffer pool internally. When an ONNXConstant
operation is reached at the first time, we read its DenseElementsAttr and store
to an array buffer. A unique buffer ID is used to map the ONNXConstantOp to
the buffer. All constant computations are done on array buffers, which is to
avoid creating DenseElementsAttr for intermediate ONNXConstantOps.
Buffers are automatically freed if they are not used.

We provide three helper functions to use when working with buffers:
1. getArrayFromAttributeOrBuffer(PatternRewriter &rewriter, Operation *op)
   - create a buffer from a dense attribute at the first time we reach the
     const 'op' and add the buffer to the buffer pool, or
   - get the buffer from the buffer pool if it was created.
2. createConstantOpAndStoreBufferPtr(..., char *buffer)
   - create a new ONNXConstantOp using the given buffer, and
   - add the buffer to the buffer pool.
3. allocateBufferFor(Value value, bool useMaxSize = false)
   - create a new buffer whose size is obtained from the type of 'value'.

Note that:
  - The buffers in the buffer pool will be automatically freed. Users don't
    need to take care about that.
  - If we create a buffer and do not put it on the buffer pool, please
    make sure that it is correctly freed. We don't recommend to do this.
    
## Write rules for constant propagation

We use MLIR declarative rewriting rules (DRR) to write patterns for constant
propagation. The DRR definition used for defining patterns is shown below:
```
class Pattern<
   dag sourcePattern,
   list<dag> resultPatterns,
   list<dag> additionalConstraints = [],
   dag benefitsAdded = (addBenefit 0)
>;
```
More information about DRR can be found [here](https://mlir.llvm.org/docs/DeclarativeRewrites/).

There is a limitation in writing DRRs for `--constprop-onnx` pass so that the compilation memory is
minimized.
- Never use ONNXConstant in the result patterns of a DRR.
 
In this section, we use an example that adds constant propagation for
ONNXAddOp.

### Step 1: Write declarative rewriting rules.
- Add a pattern to `src/Transform/ONNX/ConstProp.td`, 

```mlir
// Constant Propagation for Add
def AddConstProp : Pat<
    // source patten: From add(lhs, rhs).
    (ONNXAddOp:$addOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // result pattern: To c = lhs + rhs 
    (CreateAddOfTwoConst $addOp, $lhs, $rhs),
    // Additional constraints: if both lhs and rhs are dense constants.
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs)]>;
```
The above pattern will replace an ONNXAddOp whose inputs are constants
by a new constant by adding the inputs at compile time. To check if an input is
a constant, using ONNXConstantOp is not enough since the constant tensor can be
sparse and we now support dense constant tensors only. We need additionallly
check a dense constant tensor by using `IsFromDenseONNXConstantOp`.

In the result pattern, to produce a ONNXConstantOp, we will add `lhs`
and `rhs` at compile time, and emit an ONNXConstantOp. To minimize the
memory footprint, **this ONNXConstant does not have a DenseElementsAttr**, but
refers to an internal buffer where the real data is. DenseElementsAttrs will be
added to only **the final ONNXConstantOps of the whold pass**,
not to **intermediate` generated ONNXConstantOps.

Function `CreateAddOfTwoConst` will do the addition at compile time and return an
ONNXConstant.
```
def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;
```

### Step 2: Prepare array buffers for inputs and result 

Function `CreateAddOfTwoConst` in the pattern calls `ConstPropElementwiseBinary` whose content
is as follows.

```c++
template <typename ElementwiseBinaryOp>
ONNXConstantOp ConstPropElementwiseBinary(
    PatternRewriter &rewriter, Value replacingValue, Value lhs, Value rhs) {
  Type elementType =
      replacingValue.getType().cast<ShapedType>().getElementType();
  ArrayRef<int64_t> lhsShape = lhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape = rhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> outputShape =
      replacingValue.getType().cast<ShapedType>().getShape();
  // Get lhs and rhs values.
  char *lhsArray = getArrayFromAttributeOrBuffer(rewriter, lhs.getDefiningOp());
  char *rhsArray = getArrayFromAttributeOrBuffer(rewriter, rhs.getDefiningOp());

  // Allocate a buffer for the result.
  // Use maximum size (double or int64_t) to avoid the precision loss.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);
      
  // Do calculation.
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    IterateConstPropElementwiseBinary<ElementwiseBinaryOp, double>(
        lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
  } else if (elementType.isa<IntegerType>()) {
    // Use int64_t to avoid the precision loss during computation.
    IterateConstPropElementwiseBinary<ElementwiseBinaryOp, int64_t>(
        lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
  } else
    llvm_unreachable("Unknown data type");

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);
  
  // Return the constant.
  return res;
}
```

For each constant input defined by ONNXConstantOp, we get the array buffer
associated with it by using function `getArrayFromAttributeOrBuffer`. The buffer is created at the first time we reach an ONNXConstant.

To allocate an array buffer for the result, we use function `allocateBufferFor`
with the maximum type size to avoid precision loss.

Constant computation will operate on the two input array buffers and the result
will be stored in the result array buffer. 

To construct a result ONNXConstantOp from the result array buffer, we
use function `createConstantOpAndStoreBufferPtr`. The return ONNXConstant
operation will contain a buffer id which is associated with the buffer for the
constant, which is to avoid creating DenseElementsAttr for intermediate
ONNXConstantOps.

### Do computation on array buffers

Now we describe how to do computation on array buffers.

Because array buffers is an 1D array while the orignal data layout is tensor,
to access elements we need to convert between linear access index and tensor
index. 

We prodive two helper functions for index convertsion, i.e. 
1. getAccessIndex: to get a tensor index from a linear index.
2. getLinearAccessIndex: to get a linear index from a tensor index.
 
Below is a snippet code to demonstrate how to use them.
```
  // Do computation.
  for (int64_t i = 0; i < getNumberOfElements(outputShape); ++i) {
    // Compute indices to access the output.
    std::vector<int64_t> outputIndices = getAccessIndex(i, outputStrides);

    // Compute indices to access inputs.
    SmallVector<int64_t, 4> lhsIndices(lhsRank, 0);
    SmallVector<int64_t, 4> rhsIndices(rhsRank, 0);
    if (!broadcasting) {
      for (int k = 0; k < outputRank; ++k) {
        lhsIndices[k] = outputIndices[k];
        rhsIndices[k] = outputIndices[k];
      }
    } else {
      for (int k = 0; k < outputRank; ++k) {
        // in the lhs index range.
        if (k >= outputRank - lhsRank) {
          int lhsIndex = k - outputRank + lhsRank;
          if (lhsShape[lhsIndex] == 1)
            // broadcast
            lhsIndices[lhsIndex] = 0;
          else
            lhsIndices[lhsIndex] = outputIndices[k];
        }
        // in the rhs index range.
        if (k >= outputRank - rhsRank) {
          int rhsIndex = k - outputRank + rhsRank;
          if (rhsShape[rhsIndex] == 1)
            // broadcast
            rhsIndices[rhsIndex] = 0;
          else
            rhsIndices[rhsIndex] = outputIndices[k];
        }
      }
    }

    // Calculate element-wise binary result.
    int64_t lhsOffset = getLinearAccessIndex(lhsIndices, lhsStrides);
    int64_t rhsOffset = getLinearAccessIndex(rhsIndices, rhsStrides);

    T lhsValue = *(lhsArray + lhsOffset);
    T rhsValue = *(rhsArray + rhsOffset);
    *(resArray + i) = ComputeConstPropElementwiseBinary<ElementwiseBinaryOp, T>(
        lhsValue, rhsValue);
  }
```
The above code iterates over the linear index space of the output. For each
index, it computes a tensor index, and uses the tensor index to computes tensor
indices for the lhs and rhs according to the broadcasting rule. After that, it
computes linear indices for the lhs and rhs, then get lhs and rhs values for
addition. The result is finally stored to the result array buffer.
