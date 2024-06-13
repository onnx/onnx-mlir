# Constant Propagation for ONNX operations

This document describes `--constprop-onnx` pass which is used to do
constant propagation for operations in the ONNX dialect.

[source
code](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.td).

## Example
Given the following code:
```mlir
func @foo() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  %3 = "onnx.Constant"() {value = dense<[3.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %4 = "onnx.Add"(%2, %3) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "std.return"(%4) : (tensor<1xf32>) -> ()
}
```

If we call `onnx-mlir-op --constprop-onnx`, we will get:
```mlir
func @foo() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[6.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  "std.return"(%0) : (tensor<1xf32>) -> ()
}
```

## Remark

ONNXConstantOp uses MLIR DenseElementsAttr to store constant values. It is
important to note that, once a DenseElementsAttr is created, it is alive and
consumes memory until the end of compilation. In [Example](#example), all the
three DenseElementsAttrs in the three ONNXConstantOps exist until the end of
compilation. Especially, two intermediate DenseElementsAttrs in the two
ONNXConstantOps produced by folding the two ONNXAddOps also exist. For a
real world model, the number of intermediate DenseElementsAttrs will increase
quickly, which leads to a large memory footprint during compilation.

To avoid creating too many DenseElementsAttrs for intermediate ONNXConstantOps
during `--constprop-onnx`, we designed a mechanism that dynamically allocates and
deallocates buffers for intermediate ONNXConstantOps and only creates
DenseElementsAttr after constant propagation and other ONNX dialect passes,
just before lowering to Krnl (or any other target dialect).

This is accomplished with a custom attribute DisposableElementsAttr which
acts as a substitute for DenseElementsAttr for the common case of
non-complex scalar element types: bool and integer and floating point types.
DisposableElementsAttr implements the same ElementsAttr interface as
DenseElementsAttr and in most cases they are functionally identical and
the surrounding code doesn't need to distinguish. It just needs to use the
OnnxElementsAttrBuilder class and ElementsAttrHelper functions to
construct and access ElementsAttr instances to reap the the memory footprint
and performance benefits.

The deallocation of DisposableElementsAttr buffers happens between compiler
passes in DisposableGarbageCollector, which is run by the PassManager
between "module" passes (which are guaranteed to "stop the world" with no
other passes executing in parallel) as an "instrumentation".

DisposableElementsAttr offers other memory and speed benefits which are
outlined in the comments in the class source file and are
explained in the presentation from November 2022, linked from the
[meeting wiki page](https://github.com/onnx/onnx-mlir/wiki/Informal-meeting-agenda-and-notes#nov-29th).

## Write rules for constant propagation

We use MLIR declarative rewriting rules (DRR) to write patterns for constant
propagation. The DRR definition used for defining patterns is shown below:
```
class Pattern<
   dag sourcePattern,
   list<dag> resultPatterns,
   list<dag> additionalConstraints = [],
   list<dag> supplementalPatterns = [],
   dag benefitsAdded = (addBenefit 0)
>;
```

More information about DRR can be found [here](https://mlir.llvm.org/docs/DeclarativeRewrites/).

Now, we go through a simple example that adds constant propagation for ONNXAddOp.

### Step 1: Write DRR patterns <a id="step1"></a>

We first add a pattern to
[ConstProp.td](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.td).

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
memory footprint, this ONNXConstantOp has a DisposableElementsAttr instead of a conventional DenseElementsAttr.

Function `CreateAddOfTwoConst` will do the addition at compile time and return
an ONNXConstantOp.

```
def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;
```

### Step 2: Prepare array buffers for inputs and result <a id="step2"></a>

Function `CreateAddOfTwoConst` in the pattern calls
`ConstPropElementwiseBinary` in [ConstProp.cpp](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.cpp) whose content is as follows.

```c++
template <typename ElementwiseBinaryOp>
Value ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value replacingValue, Value lhsValue, Value rhsValue) {
  ConstPropCounters::count("ElementwiseBinary", {lhsValue, rhsValue});
  Type replacingType = mlir::cast<ShapedType>(replacingValue.getType());

  // Get lhs and rhs ElementsAttr from the values' defining constant ops.
  ElementsAttr lhs = getConstValueElements(lhsValue);
  ElementsAttr rhs = getConstValueElements(rhsValue);

  Type operandsElemType = lhs.getElementType();
  assert(operandsElemType == rhs.getElementType() &&
         "all element-wise binary ops have matching operands element types");
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr resultElements = elementsBuilder.combine(lhs, rhs, replacingType,
      combinerOfElementwiseBinaryOp<ElementwiseBinaryOp>(operandsElemType));

  // Construct and return a new ONNXConstantOp with the resultElements attribute.
  return createReplacingConstantOp(rewriter, replacingValue, resultElements)
      .getResult();
}
```
where `OnnxElementsAttrBuilder.combine(...)` broadcasts the lhs and rhs elements,
as needed, and constructs a new (Disposable) ElementsAttr whose elements are the
result of element-wise application of the binary function
`combinerOfElementwiseBinaryOp<ElementwiseBinaryOp>(operandsElemType)`
which maps the ElementwiseBinaryOp ONNX op to a c++ operator.

### TODO: Describe how to add OnnxElementsAttrBuilder builder methods for new ops

For more information about constant propagation, please see [ConstProp.td](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.td)
and
[ConstProp.cpp](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.cpp).
